#include "vocgputrain.h"
#include <QDir>
#include <QXmlSimpleReader>
#include <QXmlInputSource>
#include <QMap>

#include <fstream>

#include "gpumat.h"
#include "gpu_mlp.h"
#include "matops.h"

const QString path_annotations("Annotations");
const QString path_images("JPEGImages");

const int W = 448;
const int K = 7;
const int Classes = 30;
const int Boxes = 2;

const int cnv_size = 5;
const int mlp_size = 3;

const int cnv_do_back_layers = 1;
const int lrs = 4;

const int first_classes = 0;
const int last_classes = first_classes + K * K - 1;
const int first_boxes = last_classes + 1;
const int last_boxes = first_boxes + K * K - 1;
const int first_confidences = last_boxes + 1;
const int last_confidences = first_confidences + K * K - 1;

VOCGpuTrain::VOCGpuTrain()
{
	m_currentAn = 0;
	m_curObj = 0;
	m_refSize = 0;
	m_index = -1;
	m_check_count = 600;

	m_modelSave = "model_voc.bin";

	m_passes = 100000;
	m_batch = 5;
	m_lr = 0.00001;
	m_num_save_pass = 300;

	m_out_features = 0;
	for(int i = 0; i < K * K; ++i){
		m_cols.push_back(Classes);
		m_out_features += Classes;
	}
	for(int i = 0; i < K * K; ++i){
		m_cols.push_back(4 * Boxes);
		m_out_features += (4 * Boxes);
	}
	for(int i = 0; i < K * K; ++i){
		m_cols.push_back(Boxes);
		m_out_features += Boxes;
	}
	printf("Output features: %d, Output matrices: %d\n", m_out_features, m_cols.size());
}

bool VOCGpuTrain::setVocFolder(const QString sdir)
{
	if(sdir.isEmpty())
		return false;

	QDir dir;
	dir.setPath(sdir);

	if(!dir.exists()){
		printf("path not exists\n");
		return false;
	}

	dir.setPath(sdir + "/" + path_annotations);
	dir.setNameFilters(QStringList("*.xml"));

	if(!dir.exists()){
		printf("annotations not exists\n");
		return false;
	}

	QMap< std::string, int> map;

	m_annotations.clear();
	for(int i = 0; i < dir.count(); ++i){
		QString path = dir.path() + "/" + dir[i];
		Annotation an;
		printf("FILE: %s", path.toLatin1().data());
		if(load_annotation(path, an)){
			m_annotations.push_back(an);

			for(int i = 0; i < an.objs.size(); ++i){
				int k = map[an.objs[i].name];
				map[an.objs[i].name] = k + 1;
			}
			std::cout << "(" << an.size.width << ", " << an.size.height << ")";
		}
		std::cout << std::endl;
	}

	m_vocdir = sdir;

	printf("-----------------\n");

	int id = 1;
	for(QMap< std::string, int>::iterator it = map.begin(); it != map.end(); ++it){
		std::string name = it.key();
		int cnt = it.value();
		printf("count(%s) = %d\n", name.c_str(), cnt);
		m_classes[name] = id++;
	}
	printf("CLASSES: %d\n", map.size());

	printf("COUNT: %d\n", (int)m_annotations.size());
	return true;
}

void fillP(cv::Mat& im, cv::Rect& rec, cv::Scalar col = cv::Scalar(0, 0, 200, 0))
{
	int D = W / K;
	int bxs = (rec.x * K) / W;
	int bxe = ((rec.x + rec.width) * K) / W;
	int bys = (rec.y * K) / W;
	int bye = ((rec.y + rec.height) * K) / W;
	int cx = rec.x + rec.width/2;
	int cy = rec.y + rec.height/2;
	int bx = cx/K;
	int by = cy/K;

	std::vector<float> P[K][K];

	for(int y = bys; y <= bye; ++y){
		for(int x = bxs; x <= bxe; ++x){

			int x1 = x * D;
			int x2 = x1 + D;
			int y1 = y * D;
			int y2 = y1 + D;

			int xr = rec.x + rec.width;
			int yr = rec.y + rec.height;

			if(rec.x > x1 && rec.x < x2)x1 = rec.x;
			if(rec.y > y1 && rec.y < y2)y1 = rec.y;
			if(xr > x1 && xr < x2)x2 = xr;
			if(yr > y1 && yr < y2)y2 = yr;

			cv::Rect rt = cv::Rect(x1, y1, x2 - x1, y2 - y1);

			float p = (float)rt.width * rt.height / (D * D);
			if(x == bx && y == by){
				p = 1;
			}

			cv::rectangle(im, rt,
						  cv::Scalar(0, 200, 0), 2);
			//cv::Rect roi(x * D, y * D, D, D);
			cv::Mat _fill = im(rt);
			cv::Mat color(_fill.size(), _fill.type(), cv::Scalar(0, 0 + p * 200, 0));
			cv::addWeighted(color, 0.5, _fill, 0.5, 0, _fill);
		}
	}
}

bool VOCGpuTrain::show(int index, bool flip, const std::string name)
{
	if(!m_annotations.size())
		return false;
	if(index > m_annotations.size()){
		index = m_annotations.size() - 1;
	}

	if(m_sample.empty() || m_index != index){
		m_index = index;
		QString path = m_vocdir + "/" + path_images + "/";
		path += m_annotations[index].filename.c_str();

		if(!QFile::exists(path))
			return false;

		m_sample = cv::imread(path.toStdString());
		if(m_sample.empty())
			return false;
	}

	Annotation& it = m_annotations[index];

	cv::Mat out;
	m_sample.copyTo(out);

	if(flip){
		cv::flip(out, out, 1);
	}

	const int W = 448;

	cv::Mat out2;
	cv::Size size(W, W);
	cv::resize(m_sample, out2, size);
	if(flip){
		cv::flip(out2, out2, 1);
	}
	cv::cvtColor(out2, out2, CV_RGB2RGBA);

	int K = 7;

	int D = W / K;

	for(int i = 0; i < it.objs.size(); ++i){
		cv::Rect rec = it.objs[i].rects;
		if(flip){
			rec.x = out.cols - rec.x - rec.width;
		}
		cv::rectangle(out, rec, cv::Scalar(0, 0, 255), 1);

		std::stringstream ss;
		ss << i << " " << it.objs[i].name;
		cv::putText(out, ss.str(), rec.tl(), 1, 1, cv::Scalar(0, 0, 255), 1);

//		rec = it.objs[i].rects;
//		if(flip){
//			rec.x = out.cols - rec.x - rec.width;
//		}
		rec.x = (float)rec.x / m_sample.cols * W;
		rec.y = (float)rec.y / m_sample.rows * W;
		rec.width = (float)rec.width / m_sample.cols * W;
		rec.height = (float)rec.height / m_sample.rows * W;

//		cv::rectangle(out2, rec, cv::Scalar(0, 0, 255), 1);

		float cx = (float)(rec.x + rec.width/2) / W;
		float cy = (float)(rec.y + rec.height/2) / W;
		float dw = (float)rec.width / W;
		float dh = (float)rec.height / W;

		int bx = cx * K;
		int by = cy * K;

		cv::Rect roi = cv::Rect((bx * W)/K, (by * W)/K, W / K, W / K);

		if(roi.x > W || roi.y > W || roi.x + roi.width > W || roi.y + roi.height > W){
			printf("oops. %d: (%d, %d), (%d, %d)", index, roi.x, roi.y, roi.width, roi.height);
			continue;
		}

//		cv::Mat _fill = out2(roi);
//		cv::Mat color(_fill.size(), _fill.type(), cv::Scalar(0, 0, 200, 0));
//		cv::addWeighted(color, 0.5, _fill, 0.5, 0, _fill);
		//cv::rectangle(out2, cv::Rect(bx * W, by * W, W / K, W / K), cv::Scalar(0, 0, 255, 120), 1);

		fillP(out2, rec, cv::Scalar(0, 200, 0));

		cv::circle(out2, cv::Point(cx * W, cy * W), 4, cv::Scalar(0, 200, 0), 1);

		QString str = QString("(%1, %2), (%3, %4)").arg(cx).arg(cy).arg(dw).arg(dh);
		cv::putText(out2, str.toStdString(), rec.tl(), 1, 1, cv::Scalar(0));
	}

	for(int i = 0; i < K; ++i){
		int X = i * D;
		cv::line(out2, cv::Point(0, X), cv::Point(out2.cols, X), cv::Scalar(255, 150, 0), 1);
		cv::line(out2, cv::Point(X, 0), cv::Point(X, out2.rows), cv::Scalar(255, 150, 0), 1);
	}

	cv::imshow(name, out);
	cv::imshow(name + "2", out2);

	return true;
}

void getP(cv::Rect& rec, std::vector< ct::Matf > &classes, int cls, int bxid, const cv::Rect2f& Bx, int row = 0)
{
	if(classes.empty())
		return;

	int D = W / K;
	int bxs = (rec.x * K) / W;
	int bxe = ((rec.x + rec.width) * K) / W;
	int bys = (rec.y * K) / W;
	int bye = ((rec.y + rec.height) * K) / W;
	int cx = rec.x + rec.width/2;
	int cy = rec.y + rec.height/2;
	int bx = (cx * K)/W;
	int by = (cy * K)/W;

	if(bxid >= Boxes)
		throw new std::string("oops. some error!!!");

	std::vector<float> P[K][K];

	if(bxe == K)bxe = K - 1;
	if(bye == K)bye = K - 1;

	for(int y = bys; y <= bye; ++y){
		for(int x = bxs; x <= bxe; ++x){

			int x1 = x * D;
			int x2 = x1 + D;
			int y1 = y * D;
			int y2 = y1 + D;

			int xr = rec.x + rec.width;
			int yr = rec.y + rec.height;

			if(rec.x > x1 && rec.x < x2)x1 = rec.x;
			if(rec.y > y1 && rec.y < y2)y1 = rec.y;
			if(xr > x1 && xr < x2)x2 = xr;
			if(yr > y1 && yr < y2)y2 = yr;

			cv::Rect rt = cv::Rect(x1, y1, x2 - x1, y2 - y1);

			int off = y * K + x;

			float p = (float)rt.width * rt.height / (D * D);
			if(x == bx && y == by){
				p = 1;
				float *dB = classes[first_boxes + off].ptr(row);
				dB[bxid * 4 + 0] = Bx.x;
				dB[bxid * 4 + 1] = Bx.y;
				dB[bxid * 4 + 2] = Bx.width;
				dB[bxid * 4 + 3] = Bx.height;
			}
			float *dC = classes[first_classes + off].ptr(row);
			dC[cls] = 1;

//			float *dB = classes[first_boxes + off].ptr(row);
//			dB[bxid * 4 + 0] = Bx.x;
//			dB[bxid * 4 + 1] = Bx.y;
//			dB[bxid * 4 + 2] = Bx.width;
//			dB[bxid * 4 + 3] = Bx.height;

			float *dCf = classes[first_confidences + off].ptr(row);
			dCf[bxid] = p;
//			cv::rectangle(im, rt,
//						  cv::Scalar(0, 200, 0), 2);
//			//cv::Rect roi(x * D, y * D, D, D);
//			cv::Mat _fill = im(rt);
//			cv::Mat color(_fill.size(), _fill.type(), cv::Scalar(0, 0 + p * 200, 0));
//			cv::addWeighted(color, 0.5, _fill, 0.5, 0, _fill);
		}
	}
}

Annotation& VOCGpuTrain::getGroundTruthMat(int index, int boxes, std::vector< ct::Matf >& images,
									std::vector<ct::Matf> &res, int row, int rows, bool flip)
{
	if(index < 0 || index > m_annotations.size()){
		throw;
	}
	Annotation& it = m_annotations[index];

	int D = W / K;

	int cnt_cls = K * K;
	int cnt_bxs = K * K;
	int cnt_cnfd = K * K;
	int all_cnt = cnt_cls + cnt_bxs + cnt_cnfd;

	int id1 = 0;
	int cnt1 = cnt_cls;
	int id2 = id1 + cnt_cls;
	int cnt2 = cnt_bxs;
	int id3 = id2 + cnt2;
	int cnt3 = cnt_cnfd;

	if(res.size() != all_cnt)
		res.resize(all_cnt);
	for(int i = id1; i < id1 + cnt1; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, Classes);
		res[i].ptr(row)[0] = 0;
	}
	for(int i = id2; i < id2 + cnt2; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 4 * boxes);
		for(int j = 0; j < 4 * boxes; ++j){
			res[i].ptr(row)[j] = 0;
		}
	}
	for(int i = id3; i < id3 + cnt3; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 1 * boxes);
		for(int j = 0; j < 1 * boxes; ++j){
			res[i].ptr(row)[j] = 0;
		}
	}

	if(images.size() != rows){
		images.resize(rows);
	}
	QString path_image = m_vocdir + "/";
	path_image += path_images + "/" + it.filename.c_str();
	getImage(path_image.toStdString(), images[row], flip);

	std::vector< int > bxs;
	bxs.resize(K * K);
	for(int i = 0; i < it.objs.size(); ++i){
		cv::Rect rec = it.objs[i].rects;
		std::string name = it.objs[i].name;
		int cls = m_classes[name];

		if(flip){
			rec.x = it.size.width - rec.x - rec.width;
		}

		rec.x = (float)rec.x / it.size.width * W;
		rec.y = (float)rec.y / it.size.height * W;
		rec.width = (float)rec.width / it.size.width * W;
		rec.height = (float)rec.height / it.size.height * W;

		float cx = (float)(rec.x + rec.width/2) / W;
		float cy = (float)(rec.y + rec.height/2) / W;
		float dw = (float)rec.width / W;
		float dh = (float)rec.height / W;

		int bx = cx * K;
		int by = cy * K;


		int off = by * K + bx;
		int bi = bxs[off];
		if(bi >= boxes){
			continue;
		}

		cv::Rect2f Bx;
		Bx.x = (cx * W - bx * D) / D;
		Bx.y = (cy * W - by * D) / D;
		Bx.width = sqrtf(std::abs(dw));
		Bx.height = sqrtf(std::abs(dh));

		getP(rec, res, cls, bi, Bx, row);

		bxs[off] = bi + 1;
	}

	for(int i = first_classes; i < last_classes + 1; ++i){
		ct::Matf& m = res[first_classes + i];
		float* dC = m.ptr(row);
		int cnt = 0;
		for(int j = 0; j < Classes; ++j){
			if(dC[j] > 0)
				cnt++;
		}
		if(cnt){
			for(int j = 0; j < Classes; ++j){
				dC[j] /= cnt;
			}
		}else{
			dC[0] = 1;
//			float *dB = res[first_boxes + i].ptr(row);
//			dB[0] = dB[1] = dB[4] = dB[5] = 0.5;
//			dB[2] = dB[3] = dB[6] = dB[7] = 1;
//			float *dCf = res[first_confidences + i].ptr(row);
//			dCf[0] = dCf[1] = 1;
		}
	}

	return it;
}

void VOCGpuTrain::getGroundTruthMat(std::vector<int> indices, int boxes,
									std::vector<ct::Matf> &images, std::vector<ct::Matf> &res, bool flip)
{
	int cnt_cls = K * K;
	int cnt_bxs = K * K;
	int cnt_cnfd = K * K;
	int all_cnt = cnt_cls + cnt_bxs + cnt_cnfd;

	int id1 = 0;
	int cnt1 = cnt_cls;
	int id2 = id1 + cnt_cls;
	int cnt2 = cnt_bxs;
	int id3 = id2 + cnt2;
	int cnt3 = cnt_cnfd;

	int rows = indices.size();

	std::vector< int > flips;
	if(flip){
		flips.resize(indices.size());
		cv::randu(flips, 0, 1);
	}else{
		flips.resize(indices.size(), 0);
	}

	if(res.size() != all_cnt)
		res.resize(all_cnt);
	for(int i = id1; i < id1 + cnt1; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, Classes);
		res[i].fill(0);
	}
	for(int i = id2; i < id2 + cnt2; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 4 * boxes);
		res[i].fill(0);
	}
	for(int i = id3; i < id3 + cnt3; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 1 * boxes);
		res[i].fill(0);
	}
	if(images.size() != rows)
		images.resize(rows);


	for(int i = 0; i < indices.size(); ++i){
		getGroundTruthMat(indices[i], boxes, images, res, i, rows, flips[i]);

//		if(!res.size()){
//			res.resize(r.size());
//		}

//		for(int i = 0; i < r.size(); ++i){
//			cv::Mat& m = res[i];
//			if(m.empty()){
//				m = r[i];
//			}else{
//				m.push_back(r[i]);
//			}
//		}
	}
}

void VOCGpuTrain::getImage(const std::string &filename, ct::Matf &res, bool flip)
{
	cv::Mat m = cv::imread(filename);
	if(m.empty())
		return;
	cv::resize(m, m, cv::Size(W, W));
//	m = GetSquareImage(m, ImReader::IM_WIDTH);

	if(flip){
		cv::flip(m, m, 1);
	}

//	cv::imwrite("ss.bmp", m);

	m.convertTo(m, CV_32F, 1./255., 0);

	res.setSize(1, m.cols * m.rows * m.channels());

	int idx = 0;
	float* dX1 = res.ptr() + 0 * m.rows * m.cols;
	float* dX2 = res.ptr() + 1 * m.rows * m.cols;
	float* dX3 = res.ptr() + 2 * m.rows * m.cols;

	for(int y = 0; y < m.rows; ++y){
		float *v = m.ptr<float>(y);
		for(int x = 0; x < m.cols; ++x, ++idx){
			dX1[idx] = v[x * m.channels() + 0];
			dX2[idx] = v[x * m.channels() + 1];
			dX3[idx] = v[x * m.channels() + 2];
		}
	}
}

void VOCGpuTrain::init()
{
	m_conv.resize(cnv_size);

	m_conv[0].init(ct::Size(W, W), 3, 4, 64, ct::Size(7, 7), true, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 1, 256, ct::Size(5, 5), true);
	m_conv[2].init(m_conv[1].szOut(), 256, 1, 512, ct::Size(3, 3), true);
	m_conv[3].init(m_conv[2].szOut(), 512, 1, 1024, ct::Size(3, 3), true);
	m_conv[4].init(m_conv[3].szOut(), 1024, 1, 1024, ct::Size(3, 3), false);

	int outFeatures = m_conv.back().outputFeatures();

	m_mlp.resize(mlp_size);

	m_mlp[0].init(outFeatures, 4096, gpumat::GPU_FLOAT);
	m_mlp[1].init(4096, 2048, gpumat::GPU_FLOAT);
	m_mlp[2].init(2048, m_out_features, gpumat::GPU_FLOAT);

	m_optim.init(m_mlp);
	m_optim.setAlpha(m_lr);
}

void VOCGpuTrain::forward(std::vector<gpumat::GpuMat> &X, std::vector<gpumat::GpuMat> *pY)
{
	if(X.empty() || m_conv.empty() || m_mlp.empty())
		return;

	using namespace gpumat;

	std::vector< GpuMat > *pX = &X;

	for(size_t i = 0; i < m_conv.size(); ++i){
		conv2::convnn_gpu& cnv = m_conv[i];
		cnv.forward(pX, RELU);
		pX = &cnv.XOut();
	}

	conv2::vec2mat(m_conv.back().XOut(), m_vec2mat);

	GpuMat *pX2 = &m_vec2mat;

	etypefunction func = RELU;

	for(size_t i = 0; i < m_mlp.size(); ++i){
		if(i == m_mlp.size() - 1)
			func = LINEAR;
		mlp& _mlp = m_mlp[i];
		_mlp.forward(pX2, func);
		pX2 = &_mlp.Y();
	}

	hsplit2(*pX2, m_cols, *pY);

	m_partZ.resize(K * K);

	for(int i = first_classes; i < last_classes + 1; ++i){
		softmax((*pY)[i], 1, m_partZ[i]);
	}
	for(int i = first_confidences; i < last_confidences + 1; ++i){
		sigmoid((*pY)[i]);
	}
}

void VOCGpuTrain::backward(std::vector<gpumat::GpuMat> &pY)
{
	gpumat::hconcat2(pY, m_D);

	gpumat::GpuMat *pD = &m_D;
	for(int i = m_mlp.size() - 1; i > -1; i--){
		gpumat::mlp& mlp = m_mlp[i];
		mlp.backward(*pD, i == 0 && cnv_do_back_layers == 0);
		pD = &mlp.DltA0;
	}
	m_optim.pass(m_mlp);

	if(cnv_do_back_layers > 0){
		gpumat::mlp& mlp0 = m_mlp.front();
		gpumat::conv2::convnn_gpu& cnvl = m_conv.back();
		gpumat::conv2::mat2vec(mlp0.DltA0, cnvl.szK, m_delta_cnv);
		std::vector< gpumat::GpuMat > *pCnv = &m_delta_cnv;
		for(int i = m_conv.size() - 1; i > lrs; --i){
			gpumat::conv2::convnn_gpu& cnvl = m_conv[i];

		}
	}
}

void VOCGpuTrain::predict(std::vector<gpumat::GpuMat> &pY, std::vector<std::vector<Obj> > &res, int boxes)
{
	std::vector< ct::Matf > mY;
	std::for_each(pY.begin(), pY.end(), [&mY](const gpumat::GpuMat& it){
		mY.resize(mY.size() + 1);
		gpumat::convert_to_mat(it, mY.back());
	});
	predict(mY, res, boxes);
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
	   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

template< typename T >
void crop_sort_classes(std::vector< T >& values, int crop)
{
	std::vector< size_t > idx;
	for(auto a: sort_indexes(values)){
		idx.push_back(a);
	}
	int k = 0;
	for(auto a: idx){
		if(k++ > crop){
			values[a] = 0;
		}
	}
}

std::string get_name(QMap< std::string, int >& classes, int cls)
{
	for(QMap< std::string, int >::iterator it = classes.begin(); it != classes.end(); ++it){
		if (it.value() == cls){
			return it.key();
		}
	}
	return "";
}

void VOCGpuTrain::predict(std::vector<ct::Matf> &pY, std::vector< std::vector<Obj> > &res, int boxes)
{
	const int Crop = 5;

	int rows = pY[0].rows;

	struct IObj{
		int cls;
		float p;
	};

	res.resize(rows);

	std::vector< ct::Matf > P;
	std::vector<  IObj > iobj;
	P.resize(K * K * boxes);
	for(int i = 0; i < K * K; ++i){
		for(int b = 0; b < boxes; ++b){
			ct::v_mulColumns(pY[first_classes + i], pY[first_confidences + i], P[i * boxes + b], b);
			ct::v_cropValues<float>(P[i * boxes + b], 0.1);
		}
	}
	for(int i = 0; i < rows; ++i){
		iobj.clear();
		iobj.resize(P.size());
		for(int j = 0; j < Classes; ++j){
			std::vector<float> line;
			for(int k = 0; k < P.size(); ++k){
				float *dP = P[k].ptr(i);
				float c = dP[k];
				line.push_back(c);
			}
			crop_sort_classes(line, Crop);
			for(int k = 0; k < P.size(); ++k){
				float *dP = P[k].ptr(i);
				dP[k] = line[k];
			}
		}
		for(int j = 0; j < P.size(); ++j){
			ct::Matf& Pj = P[j];
			float *dP = Pj.ptr(i);

			IObj o1;
			o1.cls = 0;
			o1.p = dP[0];
			bool f = false;
			for(int k = 1; k < Classes; ++k){
				if(o1.p < dP[k]){
					o1.cls = k;
					o1.p = dP[k];
					printf("%d, %f, %s\n", k, o1.p, get_name(m_classes, o1.cls).c_str());
					f = true;

				}
			}
			iobj[j] = o1;
		}

		for(int k = 0; k < iobj.size(); ++k){
			IObj& io = iobj[k];
			if(io.p > 0 && io.cls > 0){
				int off1 = k / Boxes;
				int off2 = k - off1 * Boxes;
				ct::Matf& B = pY[off1 + first_boxes];
				float *dB = B.ptr(i);
				Obj obj;

				cv::Rect rec;

				int offy = off1 / K;
				int offx = off1 - offy * K;
				float D = W / K;

				rec.x = offx * D + dB[off2 * 4 + 0] * D;
				rec.y = offy * D + dB[off2 * 4 + 1] * D;
				rec.width = dB[off2 * 4 + 2] * W;
				rec.height = dB[off2 * 4 + 3] * W;

				obj.name = get_name(m_classes, io.cls);
				obj.rects = rec;
				obj.p = io.p;
				res[i].push_back(obj);
			}
		}

	}
}

int VOCGpuTrain::passes() const
{
	return m_passes;
}

void VOCGpuTrain::setPasses(int passes)
{
	m_passes = passes;
}

int VOCGpuTrain::batch() const
{
	return m_batch;
}

void VOCGpuTrain::setBatch(int batch)
{
	m_batch = batch;
}

float VOCGpuTrain::lr() const
{
	return m_lr;
}

void VOCGpuTrain::setLerningRate(float lr)
{
	m_lr = lr;

	m_optim.setAlpha(lr);
}

int VOCGpuTrain::numSavePass() const
{
	return m_num_save_pass;
}

void VOCGpuTrain::setNumSavePass(int num)
{
	m_num_save_pass = num;
}

void VOCGpuTrain::setSeed(int seed)
{
	cv::setRNGSeed(seed);
}

void cnv2gpu(std::vector< ct::Matf >& In, std::vector< gpumat::GpuMat >& Out)
{
	Out.resize(In.size());
	for(size_t i = 0; i < In.size(); ++i){
		gpumat::convert_to_gpu(In[i], Out[i]);
	}
}

void get_delta(std::vector< gpumat::GpuMat >& t, std::vector< gpumat::GpuMat >& y, double lambda = 1., bool test = false)
{
	for(int i = first_classes; i < last_classes + 1; ++i){
		if(test){
			gpumat::save_gmat(t[i], "test/cls" + std::to_string(i));
			gpumat::save_gmat(y[i], "test/ycls" + std::to_string(i));
		}
		gpumat::sub(t[i], y[i], t[i]);
	}
	for(int i = first_boxes; i < last_boxes + 1; ++i){
		if(test){
			gpumat::save_gmat(t[i], "test/boxes" + std::to_string(i));
			gpumat::save_gmat(y[i], "test/ybxs" + std::to_string(i));
		}
		gpumat::sub(t[i], y[i], lambda, lambda);
	}
	for(int i = first_confidences; i < last_confidences + 1; ++i){
		if(test){
			gpumat::save_gmat(t[i], "test/cfd" + std::to_string(i));
			gpumat::save_gmat(y[i], "test/ycfd" + std::to_string(i));
		}
		gpumat::back_delta_sigmoid(t[i], y[i]);
//		gpumat::sub(t[i], y[i], t[i]);
	}
}

float get_loss(std::vector< gpumat::GpuMat >& t)
{
	ct::Matf mat;
	float res1 = 0;
	for(int i = first_classes; i < last_classes + 1; ++i){
		gpumat::elemwiseSqr(t[i], t[i]);
		gpumat::convert_to_mat(t[i], mat);
		res1 += mat.sum() / mat.rows;
	}
	res1 /= (last_classes - first_classes + 1);

	float res2 = 0;
	for(int i = first_boxes; i < last_boxes + 1; ++i){
		gpumat::elemwiseSqr(t[i], t[i]);
		gpumat::convert_to_mat(t[i], mat);
		res2 += mat.sum() / mat.rows;
	}
	res2 /= (last_boxes - first_boxes + 1);

	float res3 = 0;
	for(int i = first_confidences; i < last_confidences + 1; ++i){
		gpumat::elemwiseSqr(t[i], t[i]);
		gpumat::convert_to_mat(t[i], mat);
		res3 += mat.sum() / mat.rows;
	}
	res3 /= (last_confidences - first_confidences + 1);

	return res1 + res2 + res3;
}

void VOCGpuTrain::doPass()
{
	std::vector< ct::Matf > mX, mY;

	std::vector< gpumat::GpuMat > X;
	std::vector< gpumat::GpuMat > y, t;
	std::vector< int > cols;
	cols.resize(m_batch);
	for(int i = 0; i < m_passes; ++i){
		cv::randu(cols, 0, m_annotations.size() - 1);
		getGroundTruthMat(cols, Boxes, mX, mY, true);
		cnv2gpu(mX, X);
		cnv2gpu(mY, y);

		forward(X, &t);

		get_delta(t, y, 1., (i % 100) == 0);

		backward(t);

//		gpumat::convert_to_mat(m_mlp.back().W, m2);
//		s = m2 - m1;
//		ct::v_elemwiseSqr(s);
//		float fs = s.sum();

//		gpumat::convert_to_mat(t[0], m1);
//		ct::v_elemwiseSqr(m1);
//		float s2 = m1.sum();

		printf("pass=%d    \r", i);
		std::cout << std::flush;

		if((i % m_num_save_pass) == 0 && i > 0 || i == 30){
			int k = 0;
			float loss = 0;
			while( k < m_check_count){
				cv::randu(cols, 0, m_annotations.size() - 1);
				getGroundTruthMat(cols, Boxes, mX, mY);
				cnv2gpu(mX, X);
				cnv2gpu(mY, y);

				forward(X, &t);

				get_delta(t, y);

				loss += get_loss(t);

				k += m_batch;
			}
			loss /= m_check_count;
			printf("pass=%d, loss=%f    \n", i, loss);
			saveModel(m_modelSave);
		}
	}
}

void VOCGpuTrain::test()
{
	std::vector< ct::Matf > mX, mY;
	std::vector< std::vector< Obj > > res;

	std::vector< gpumat::GpuMat > X;
	std::vector< gpumat::GpuMat > y, t;
	std::vector< int > cols;

//	cols.resize(m_batch);

	//cv::randu(cols, 0, m_annotations.size() - 1);
//	cols.push_back(2529);
//	cols.push_back(1064);
//	cols.push_back(4569);
//	cols.push_back(177);
//	cols.push_back(907);
//	cols.push_back(2818);
//	cols.push_back(3521);
//	cols.push_back(1719);
//	cols.push_back(2728);
	cols.push_back(4810);
	getGroundTruthMat(cols, Boxes, mX, mY, true);
	cnv2gpu(mX, X);
	cnv2gpu(mY, y);

	forward(X, &t);

	predict(t, res, Boxes);
}

size_t VOCGpuTrain::size() const
{
	return m_annotations.size();
}

bool VOCGpuTrain::loadModel(const QString &model, bool load_mlp)
{
	QString n = QDir::fromNativeSeparators(model);

	std::fstream fs;
	fs.open(n.toStdString(), std::ios_base::in | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", n.toLatin1().data());
		return false;
	}

	m_model = n;

//	read_vector(fs, m_cnvlayers);
//	read_vector(fs, m_layers);

//	fs.read((char*)&m_szA0, sizeof(m_szA0));

//	setConvLayers(m_cnvlayers, m_szA0);

	init();

	int cnvs, mlps;

	/// size of convolution array
	fs.read((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.read((char*)&mlps, sizeof(mlps));

	printf("Load model: conv size %d, mlp size %d\n", cnvs, mlps);

	if(m_conv.size() < cnvs)
		m_conv.resize(cnvs);

	printf("conv\n");
	for(size_t i = 0; i < cnvs; ++i){
		gpumat::conv2::convnn_gpu &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W[0].rows, cnv.W[0].cols);
	}

	if(load_mlp){
		m_mlp.resize(mlps);
		printf("mlp\n");
		for(size_t i = 0; i < m_mlp.size(); ++i){
			gpumat::mlp &mlp = m_mlp[i];
			mlp.read2(fs);
			printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
		}
	}

	printf("model loaded.\n");
	return true;
}

void VOCGpuTrain::saveModel(const QString &name)
{
	QString n = QDir::fromNativeSeparators(name);

	std::fstream fs;
	fs.open(n.toStdString(), std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", n.toLatin1().data());
		return;
	}

//	write_vector(fs, m_cnvlayers);
//	write_vector(fs, m_layers);

//	fs.write((char*)&m_szA0, sizeof(m_szA0));

	int cnvs = m_conv.size(), mlps = m_mlp.size();

	/// size of convolution array
	fs.write((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.write((char*)&mlps, sizeof(mlps));

	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::conv2::convnn_gpu &cnv = m_conv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		gpumat::mlp& mlp = m_mlp[i];
		mlp.write2(fs);
	}

	printf("model saved.\n");
}

void VOCGpuTrain::setModelSaveName(const QString &name)
{
	m_modelSave = name;
}

bool VOCGpuTrain::load_annotation(const QString &fileName, Annotation& annotation)
{
	if(!QFile::exists(fileName))
		return false;

	QFile file(fileName);
	if(!file.open(QIODevice::ReadOnly))
		return false;

	m_currentAn = &annotation;

	QXmlSimpleReader xml;
	QXmlInputSource input(&file);
	xml.setContentHandler(this);
	bool parse = xml.parse(&input);

	file.close();

	if(!parse){
		file.close();
		return false;
	}

	return true;
}

bool VOCGpuTrain::startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &atts)
{
	m_key = qName;
	if(m_key == "object"){
		if(m_currentAn){
			m_currentAn->objs.push_back(Obj());
			m_curObj = &m_currentAn->objs.back();
		}
	}
	if(m_key == "bndbox"){
		m_box.clear();
	}
	if(m_key == "size" && m_currentAn){
		m_refSize = &m_currentAn->size;
	}
	return true;
}

bool VOCGpuTrain::endElement(const QString &namespaceURI, const QString &localName, const QString &qName)
{
	m_key = "";
	if(qName == "object"){
		m_curObj = 0;
	}
	if(qName == "bndbox" && m_curObj){
		m_curObj->rects = cv::Rect(m_box.x(), m_box.y(), m_box.w(), m_box.h());
		m_box.clear();
	}
	if(qName == "size"){
		m_refSize = 0;
	}
	return true;
}

bool VOCGpuTrain::characters(const QString &ch)
{
	if(!m_currentAn)
		return false;

	if(m_key == "folder"){
		m_currentAn->folder = ch.toStdString();
	}
	if(m_key == "filename")
		m_currentAn->filename = ch.toStdString();

	if(m_key == "width" && m_refSize){
		m_refSize->width = ch.toInt();
	}
	if(m_key == "height" && m_refSize){
		m_refSize->height = ch.toInt();
	}

	if(m_curObj){
		if(m_key == "name"){
			m_curObj->name = ch.toStdString();
		}
		if(m_key == "xmin"){
			m_box.xmin = ch.toInt();
		}
		if(m_key == "xmax"){
			m_box.xmax = ch.toInt();
		}
		if(m_key == "ymin"){
			m_box.ymin = ch.toInt();
		}
		if(m_key == "ymax"){
			m_box.ymax = ch.toInt();
		}
	}

	return true;
}

QString VOCGpuTrain::errorString() const
{
	return "error";
}
