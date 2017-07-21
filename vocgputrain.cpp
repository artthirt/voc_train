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

const int cnv_size = 4;
const int mlp_size = 3;

const int cnv_do_back_layers = 0;
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
	m_check_count = 300;

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

		cv::rectangle(out2, rec, cv::Scalar(0, 0, 255), 1);

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

		cv::Mat _fill = out2(roi);
		cv::Mat color(_fill.size(), _fill.type(), cv::Scalar(0, 0, 200, 0));
		cv::addWeighted(color, 0.5, _fill, 0.5, 0, _fill);
		//cv::rectangle(out2, cv::Rect(bx * W, by * W, W / K, W / K), cv::Scalar(0, 0, 255, 120), 1);

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
			res[i] = ct::Matf::zeros(rows, 1);
	}
	for(int i = id2; i < id2 + cnt2; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 4 * boxes);
	}
	for(int i = id3; i < id3 + cnt3; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 1 * boxes);
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

		cv::Rect roi = cv::Rect((bx * W)/K, (by * W)/K, W / K, W / K);

		if(roi.x > W || roi.y > W || roi.x + roi.width > W || roi.y + roi.height > W){
			printf("oops. %d: (%d, %d), (%d, %d)", index, roi.x, roi.y, roi.width, roi.height);
			continue;
		}

		int off = by * K + bx;
		int bi = bxs[off];
		if(bi > boxes){
			continue;
		}

		ct::Matf& Cls = res[id1 + off];
		float *dCls = Cls.ptr(row);
		dCls[0] = cls;

		ct::Matf& Bxs = res[id2 + off];
		float *dBxs = Bxs.ptr(row);
		dBxs[4 * bi + 0] = (cx * W - bx * D) / D;
		dBxs[4 * bi + 1] = (cy * W - by * D) / D;
		dBxs[4 * bi + 2] = dw;
		dBxs[4 * bi + 3] = dh;

		ct::Matf& Cnfd = res[id3 + off];
		float *dCnfd = Cnfd.ptr(row);
		dCnfd[bi] = 1.;

		bxs[off] = bi + 1;
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
			res[i] = ct::Matf::zeros(rows, 1);
	}
	for(int i = id2; i < id2 + cnt2; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 4 * boxes);
	}
	for(int i = id3; i < id3 + cnt3; ++i){
		if(res[i].empty())
			res[i] = ct::Matf::zeros(rows, 1 * boxes);
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
//	m_conv[4].init(m_conv[3].szOut(), 1024, 1, 1024, ct::Size(3, 3), false);

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
		mlp.backward(*pD, i == 0 && cnv_do_back_layers > 0);
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

void get_delta(std::vector< gpumat::GpuMat >& t, std::vector< gpumat::GpuMat >& y)
{
	for(int i = first_classes; i < last_classes + 1; ++i){
		gpumat::subIndOne(t[i], y[i], t[i]);
	}
	for(int i = first_boxes; i < last_boxes + 1; ++i){
		gpumat::sub(t[i], y[i], t[i]);
	}
	for(int i = first_confidences; i < last_confidences + 1; ++i){
//		gpumat::deriv_sigmoid(t[i], y[i]);
		gpumat::sub(t[i], y[i], t[i]);
	}
}

float get_loss(std::vector< gpumat::GpuMat >& t)
{
	ct::Matf mat;
	float res = 0;
	for(int i = first_classes; i < last_classes + 1; ++i){
		gpumat::elemwiseSqr(t[i], t[i]);
		gpumat::convert_to_mat(t[i], mat);
		res += mat.sum() / mat.rows;
	}
	for(int i = first_boxes; i < last_boxes + 1; ++i){
		gpumat::elemwiseSqr(t[i], t[i]);
		gpumat::convert_to_mat(t[i], mat);
		res += mat.sum() / mat.rows;
	}
	for(int i = first_confidences; i < last_confidences + 1; ++i){
		gpumat::elemwiseSqr(t[i], t[i]);
		gpumat::convert_to_mat(t[i], mat);
		res += mat.sum() / mat.rows;
	}
	return res;
}

void VOCGpuTrain::doPass()
{
	std::vector< ct::Matf > mX, mY;
	ct::Matf m1, m2, s;

	std::vector< gpumat::GpuMat > X;
	std::vector< gpumat::GpuMat > y, t;
	std::vector< int > cols;
	cols.resize(m_batch);
	for(int i = 0; i < m_passes; ++i){
		cv::randu(cols, 0, m_annotations.size() - 1);
		getGroundTruthMat(cols, Boxes, mX, mY, true);
		cnv2gpu(mX, X);
		cnv2gpu(mY, y);

		gpumat::convert_to_mat(m_mlp.back().W, m1);

		forward(X, &t);

		get_delta(t, y);

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

		if((i % m_num_save_pass) == 0/* && i > 0*/){
			int k = 0;
			float loss = 0;
			while( k < m_check_count){
				cv::randu(cols, 0, m_annotations.size() - 1);
				getGroundTruthMat(cols, Boxes, mX, mY, true);
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
