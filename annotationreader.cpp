#include "annotationreader.h"

#include "metaconfig.h"

#include <QDir>
#include <QFile>

#include <QDateTime>

const QString path_annotations("Annotations");
const QString path_images("JPEGImages");

AnnotationReader::AnnotationReader()
{
	m_currentAn = 0;
	m_refSize = 0;
	m_index = -1;
	m_index_im = 0;
}

bool AnnotationReader::setVocFolder(const QString sdir)
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

	annotations.clear();
	for(size_t i = 0; i < dir.count(); ++i){
		QString path = dir.path() + "/" + dir[i];
		Annotation an;
		printf("FILE: %s", path.toLatin1().data());
		if(load_annotation(path, an)){
			annotations.push_back(an);

			for(Obj obj: an.objs){
				int k = map[obj.name];
				map[obj.name] = k + 1;
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
		classes[name] = id++;
	}
	printf("CLASSES: %d\n", map.size());

	printf("COUNT: %d\n", (int)annotations.size());
	return true;
}

void AnnotationReader::set_seed(int seed)
{
	m_gt.seed(seed);
}

bool AnnotationReader::load_annotation(const QString &fileName, Annotation& annotation)
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

bool AnnotationReader::startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &atts)
{
	m_key = qName.trimmed().toLower();
	if(m_key == "object"){
		if(m_currentAn){
			m_currentAn->objs.push_back(Obj());
			m_curObj.push_back(&m_currentAn->objs.back());
		}
	}
	if(m_key == "part"){
		if(m_currentAn){
			m_currentAn->objs.push_back(Obj());
			m_curObj.push_back(&m_currentAn->objs.back());
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

bool AnnotationReader::endElement(const QString &namespaceURI, const QString &localName, const QString &qName)
{
	m_key = "";
	QString Name = qName.trimmed().toLower();
	if(Name == "object" && !m_curObj.empty()){
		m_curObj.pop_back();
	}
	if(Name == "part" && !m_curObj.empty()){
		m_curObj.pop_back();
	}
	if(Name == "bndbox" && !m_curObj.empty()){
		m_curObj.back()->rect = cv::Rect(m_box.x(), m_box.y(), m_box.w(), m_box.h());
		m_box.clear();
	}
	if(Name == "size"){
		m_refSize = 0;
	}
	return true;
}

bool AnnotationReader::characters(const QString &ch)
{
	if(!m_currentAn)
		return true;

	if(m_key == "folder"){
		m_currentAn->folder = ch.toStdString();
	}
	if(m_key == "filename"){
		m_currentAn->filename = ch.toStdString();
	}

	if(m_key == "width" && m_refSize){
		m_refSize->width = ch.toInt();
	}
	if(m_key == "height" && m_refSize){
		m_refSize->height = ch.toInt();
	}

	if(!m_curObj.empty()){

		if(m_key == "name"){
			m_curObj.back()->name = ch.toStdString();
		}
		if(m_key == "xmin"){
			m_box.xmin = ch.toFloat();
		}
		if(m_key == "xmax"){
			m_box.xmax = ch.toFloat();
		}
		if(m_key == "ymin"){
			m_box.ymin = ch.toFloat();
		}
		if(m_key == "ymax"){
			m_box.ymax = ch.toFloat();
		}
	}

	return true;
}

QString AnnotationReader::errorString() const
{
	return "error";
}

size_t AnnotationReader::size() const
{
	return annotations.size();
}

///////////////////////////////
///////////////////////////////
///////////////////////////////

void fillP(cv::Mat& im, cv::Rect& rec, int id, cv::Scalar col = cv::Scalar(0, 0, 200, 0))
{
	using namespace meta;

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

	if(bxe == K)bxe = K - 1;
	if(bye == K)bye = K - 1;

	int xr = rec.x + rec.width;
	int yr = rec.y + rec.height;

	for(int y = bys; y <= bye; ++y){
		for(int x = bxs; x <= bxe; ++x){

			int x1 = x * D;
			int x2 = x1 + D;
			int y1 = y * D;
			int y2 = y1 + D;

			if(rec.x >= x1 && rec.x < x2)x1 = rec.x;
			if(rec.y >= y1 && rec.y < y2)y1 = rec.y;
			if(xr >= x1 && xr < x2)x2 = xr;
			if(yr >= y1 && yr < y2)y2 = yr;

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
			cv::addWeighted(color, 0.2, _fill, 0.5, 0, _fill);
		}
	}
}

bool AnnotationReader::show(int index, bool flip, const std::string name)
{
	using namespace meta;

	if(!annotations.size())
		return false;
	if(index > (int)annotations.size()){
		index = annotations.size() - 1;
	}

	if(m_sample.empty() || m_index != index){
		m_index = index;
		QString path = m_vocdir + "/" + path_images + "/";
		path += annotations[index].filename.c_str();

		if(!QFile::exists(path))
			return false;

		m_sample = cv::imread(path.toStdString());
		if(m_sample.empty())
			return false;

	}

	static std::vector< ct::Matf > ims;
	std::vector< std::vector< ct::Matf > > res;
	getGroundTruthMat(index, Boxes, ims, res, 0, 1, flip, false);

	Annotation& it = annotations[index];

	cv::Mat out;
	m_sample.copyTo(out);

	if(flip){
		cv::flip(out, out, 1);
	}

//	const int W = 448;

	cv::Mat out2;
	cv::Size size(W, W);
	cv::resize(m_sample, out2, size);
	if(flip){
		cv::flip(out2, out2, 1);
	}
	cv::cvtColor(out2, out2, CV_RGB2RGBA);

//	int K = 7;

	int D = W / K;

	int id = 0;
	for(Obj obj: it.objs){
		cv::Rect rec = obj.rect;
		if(flip){
			rec.x = out.cols - rec.x - rec.width;
		}
		cv::rectangle(out, rec, cv::Scalar(0, 0, 255), 1);

		std::stringstream ss;
		ss << id << " " << obj.name;
		cv::putText(out, ss.str(), rec.tl(), 1, 1, cv::Scalar(0, 0, 255), 1);

		float dw = (float)rec.width / it.size.width;
		float dh = (float)rec.height / it.size.height;
		float cx = (float)rec.x / it.size.width + dw/2;
		float cy = (float)rec.y / it.size.height + dh/2;

		rec = cv::Rect((cx - dw/2) * W, (cy - dh/2) * W, dw * W, dh * W);

		fillP(out2, rec, id, cv::Scalar(0, 200, 0));

		QString str = QString("%1").arg(id);
		cv::putText(out2, str.toStdString(), rec.tl(), 1, 1, cv::Scalar(255, 255, 255));

		id++;
	}

//	for(size_t i = 0; i < it.objs.size(); ++i){
//		cv::Rect rec = it.objs[i].rect;
//		if(flip){
//			rec.x = out.cols - rec.x - rec.width;
//		}
//		float dw = (float)rec.width / it.size.width;
//		float dh = (float)rec.height / it.size.height;
//		float cx = (float)rec.x / it.size.width + dw/2;
//		float cy = (float)rec.y / it.size.height + dh/2;

//		cv::circle(out2, cv::Point(cx * W, cy * W), 4, cv::Scalar(255, 100, 0), 2);
//		cv::rectangle(out2, cv::Rect((cx - dw/2) * W, (cy - dh/2) * W, dw * W, dh * W), cv::Scalar(0, 0, 255), 2);
//	}

	if(!res.empty()){
		for(int k = 0; k < K * K; ++k){
			ct::Matf& m  = res[0][0];
			ct::Matf& cf = res[0][2];
			ct::Matf& bx = res[0][1];
			float *dCls = m.ptr();
			float *dCfd = cf.ptr();
			float *dBx = bx.ptr();
			int y = k / K;
			int x = k - y * K;

			cv::Point pt;
			pt.x = x * D + D/2;
			pt.y = y * D + D/2;

			for(int j = 1; j < m.cols; ++j){
				float c = dCls[j];
				if(c > 0.1){
					cv::circle(out2, pt, 4, cv::Scalar(0, 0, 255), 2);
				}
			}
			int xo = 0;
			for(int j = 0; j < cf.cols; ++j){
				float c = dCfd[j];
				if(c > 0.1){
					cv::circle(out2, pt + cv::Point(-D/2 + 10 + xo, -D/2 + 10), 3, cv::Scalar(100, 255, 0), 2);
					xo += 8;

					float dx = dBx[j * 4 + 0];
					float dy = dBx[j * 4 + 1];
					float dw = dBx[j * 4 + 2];
					float dh = dBx[j * 4 + 3];

					float xb = dx * D + x * D;
					float yb = dy * D + y * D;
					float wb = (dw * dw) * W;
					float hb = (dh * dh) * W;

					cv::Point tl(xb - wb/2, yb - hb/2);
					cv::circle(out2, cv::Point(xb, yb), 4, cv::Scalar(255, 100, 0), 2);
					cv::rectangle(out2, cv::Rect(tl.x, tl.y, wb, hb), cv::Scalar(0, 0, 255), 2);
				}
			}
		}
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

void getP(cv::Rect& rec, std::vector< ct::Matf > &classes, int cls, const cv::Rect2f& Bx,
		  std::vector< int >& bxss, int row = 0)
{
	using namespace meta;

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

//	if(bxid >= Boxes)
//		throw new std::string("oops. some error!!!");

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

			int bxid = bxss[off];
			if(bxid >= 1)
				continue;

			float p = (float)rt.width * rt.height / (D * D);
			if(x == bx && y == by){
				p = 1;
				float *dB = classes[1].ptr(row);
				dB[bxid * 4 + 0] = Bx.x;
				dB[bxid * 4 + 1] = Bx.y;
				dB[bxid * 4 + 2] = Bx.width;
				dB[bxid * 4 + 3] = Bx.height;
			}
			float *dC = classes[0].ptr(row);
			dC[cls] = 1;

//			float *dB = classes[first_boxes + off].ptr(row);
//			dB[bxid * 4 + 0] = Bx.x;
//			dB[bxid * 4 + 1] = Bx.y;
//			dB[bxid * 4 + 2] = Bx.width;
//			dB[bxid * 4 + 3] = Bx.height;

			float *dCf = classes[2].ptr(row);
			dCf[bxid] = p;

			bxss[off] = bxid + 1;
//			cv::rectangle(im, rt,
//						  cv::Scalar(0, 200, 0), 2);
//			//cv::Rect roi(x * D, y * D, D, D);
//			cv::Mat _fill = im(rt);
//			cv::Mat color(_fill.size(), _fill.type(), cv::Scalar(0, 0 + p * 200, 0));
//			cv::addWeighted(color, 0.5, _fill, 0.5, 0, _fill);
		}
	}
}

void AnnotationReader::update_output(std::vector< std::vector< ct::Matf > >& res, const Obj& ob, int off, int bxid, int row)
{
	using namespace meta;

	std::string name = ob.name;
	int cls = classes[name];

	float *dB = res[row][1].ptr(off);
	dB[bxid * 4 + 0] = (float)ob.rectf.x;
	dB[bxid * 4 + 1] = (float)ob.rectf.y;
	dB[bxid * 4 + 2] = (float)ob.rectf.width;
	dB[bxid * 4 + 3] = (float)ob.rectf.height;

	float *dC = res[row][0].ptr(off);
	dC[cls] = 1;

	float *dCf = res[row][2].ptr(off);
	dCf[bxid] = 1;

//	std::cout << "bxid " << bxid << std::endl << std::flush;
}

Annotation& AnnotationReader::getGroundTruthMat(int index, int boxes, std::vector< ct::Matf >& images,
												std::vector< std::vector< ct::Matf > > &res, int row, int rows,
												bool flip, bool load_image, bool aug, bool init_input)
{
	using namespace meta;

	if(index < 0 || index > (int)annotations.size()){
		throw;
	}
	Annotation& it = annotations[index];

	const float D = W / K;

	if(init_input){

		res.resize(rows);
		for(std::vector< ct::Matf > &v : res){
			v.resize(3);
			v[0].setSize(K * K, Classes);
			v[1].setSize(K * K, Rects);
			v[2].setSize(K * K, Boxes);
		}

	}

	{
		res[row][0].fill(0);
		res[row][1].fill(0);
		res[row][2].fill(0);
	};

	int xoff = 0, yoff = 0;

	if(aug){
		std::normal_distribution<float> nd(0, W * 0.1);
		xoff = nd(m_gt);
		yoff = nd(m_gt);
	}

	if(lambdaBxs.empty()){
		lambdaBxs.resize(rows);
	}
	if(init_input){
		lambdaBxs[row].setSize(K * K, 1);
	}
	lambdaBxs[row].fill(0.5);

	if(load_image){
		if((int)images.size() != rows){
			images.resize(rows);
		}
		QString path_image = m_vocdir + "/";
		path_image += path_images + "/" + it.filename.c_str();
		getImage(path_image.toStdString(), images[row], flip, aug, cv::Point(xoff, yoff));
	}

	std::vector< Obj > objs[K * K];

#if DEBUG_IMAGE
	cv::Mat im;
	if(load_image && images.size() > row){
		getMat(images[row], im, cv::Size(W, W));
	}
#endif

	for(Obj obj: it.objs){
		cv::Rect rec = obj.rect;
		rec.x += (xoff * it.size.width) / W;
		rec.y += (yoff * it.size.height) / W;

		if(flip){
			rec.x = it.size.width - rec.x - rec.width;
		}
		float dw = (float)rec.width / it.size.width;
		float dh = (float)rec.height / it.size.height;
		float cx = (float)rec.x / it.size.width + dw/2;
		float cy = (float)rec.y / it.size.height + dh/2;
		if(cx >= 1 || cy >= 1 || cx < 0 ||cy < 0)
			continue;

		int bx = cx * K;
		int by = cy * K;

		int off = by * K + bx;

		Obj ob = obj;
		ob.rect = cv::Rect((cx - dw/2) * W, (cy - dh/2) * W, dw * W, dh * W);

		float ccx = (float)(cx * W - bx * D) / D;
		float ccy = (float)(cy * W - by * D) / D;

		ob.rectf = cv::Rect2f(ccx, ccy, sqrtf(std::abs(dw)), sqrtf(std::abs(dh)));

		objs[off].push_back(ob);

#if DEBUG_IMAGE
		if(!im.empty()){
			rec.width = dw * W;
			rec.height = dh * W;
			rec.x = ccx * D + bx * D - rec.width/2;
			rec.y = ccy * D + by * D - rec.height/2;

			cv::rectangle(im, rec, cv::Scalar(0, 0, 255), 2);
		}
#endif
	}

#if DEBUG_IMAGE
	if(!im.empty())
		cv::imwrite("images/" + std::to_string(m_index_im) + ".jpg", im);
#endif

//	for(int i = 0; i < K * K; ++i){
//		if(objs[i].size())
//			std::cout << i << "(" << objs[i].size() << "); ";
//	}
//	std::cout << std::endl;

	for(int i = 0; i < K * K; ++i){
		size_t C = objs[i].size();

		if(C > 0){
			int off = i;

			/// sort the found objects with ascend their area
			std::sort(objs[i].begin(), objs[i].end(), [](const Obj& ob1, const Obj& ob2){
				return ob1.rectf.width * ob1.rectf.height < ob2.rectf.width * ob2.rectf.height;
			});

			/// update matrices for the found objects
			int bxid1 = -1, bxid2 = -1;
			std::for_each(objs[i].begin(), objs[i].end(), [&](const Obj& ob){
				float ar = ob.rectf.height / ob.rectf.width;
				if(ar >= 1.1){
					if(bxid1 < 0){
						bxid1 = 1;
						update_output(res, ob, off, 0, row);
					}
				}else{
					if(bxid2 < 0){
						bxid2 = 1;
						update_output(res, ob, off, 1, row);
					}
				}
			});
		}
	}

	{
		ct::Matf& m = res[row][0];
		float p = 0;

		for(int i = 0; i < K * K; ++i){
			float* dC = m.ptr(i);
			p = 0;
			for(int j = 0; j < Classes; ++j){
				p += dC[j];
			}
			if(p > 0.1){
				for(int j = 0; j < Classes; ++j) dC[j] /= p;
				lambdaBxs[row].ptr(i)[0] = 5.;
			}else{
				dC[0] = 1;
			}
		}
	}

	return it;
}

void AnnotationReader::getGroundTruthMat(std::vector<int> indices, int boxes,
										 std::vector<ct::Matf> &images, std::vector< std::vector< ct::Matf > > &res,
										 bool flip, bool aug)
{
	using namespace meta;

	int rows = indices.size();

	std::vector< int > flips;
	if(flip){
		std::binomial_distribution<int> bd(1, 0.5);
		flips.resize(indices.size());
		for(size_t i = 0; i < flips.size(); ++i){
			flips[i] = bd(m_gt);
		}
	}else{
		flips.resize(indices.size(), 0);
	}

	res.resize(rows);
	for(std::vector< ct::Matf > &v : res){
		v.resize(3);
		v[0].setSize(K * K, Classes);
		v[1].setSize(K * K, Rects);
		v[2].setSize(K * K, Boxes);
	}

	std::for_each(res.begin(), res.end(), [=](std::vector< ct::Matf >& mat){
		mat[0].fill(0);
		mat[1].fill(0);
		mat[2].fill(0);
		//mat.fill(0);
	});

	lambdaBxs.resize(rows);
	for(size_t i = 0; i < lambdaBxs.size(); ++i){
		lambdaBxs[i].setSize(K * K, 1);
		lambdaBxs[i].fill(0.5);
	}

	if((int)images.size() != rows)
		images.resize(rows);

	m_index_im = 0;
	for(size_t i = 0; i < indices.size(); ++i, ++m_index_im){
		getGroundTruthMat(indices[i], boxes, images, res, i, rows, flips[i], true, aug, false);
	}
}

void offsetImage(cv::Mat &image, cv::Scalar bordercolour, int xoffset, int yoffset)
{
	using namespace cv;
	float mdata[] = {
		1, 0, xoffset,
		0, 1, yoffset
	};

	Mat M(2, 3, CV_32F, mdata);
	warpAffine(image, image, M, image.size());
}

void AnnotationReader::getImage(const std::string &filename, ct::Matf &res, bool flip, bool aug, const cv::Point &off)
{
	using namespace meta;

	cv::Mat m = cv::imread(filename);
	if(m.empty())
		return;
	cv::resize(m, m, cv::Size(W, W));
//	m = GetSquareImage(m, ImReader::IM_WIDTH);

	if(aug && (off.x != 0 || off.y != 0)){
		offsetImage(m, cv::Scalar(0), off.x, off.y);
	}

	if(flip){
		cv::flip(m, m, 1);
	}

//	m.convertTo(m, CV_32F, 1./255., 0);
//	cv::imwrite("ss.bmp", m);
	float c1 = 1., c2 = 1., c3 = 1.;
	if(!aug){
		m.convertTo(m, CV_32F, 1./255., 0);
	}else{
		std::normal_distribution<float> nd(0, 0.1);
		float br = nd(m_gt);
		float cntr = nd(m_gt);
		m.convertTo(m, CV_32F, (0.95 + br)/255., cntr);
		c1 = 0.95 + nd(m_gt);
		c2 = 0.95 + nd(m_gt);
		c3 = 0.95 + nd(m_gt);
	}

	res.setSize(1, m.cols * m.rows * m.channels());

	float* dX1 = res.ptr() + 0 * m.rows * m.cols;
	float* dX2 = res.ptr() + 1 * m.rows * m.cols;
	float* dX3 = res.ptr() + 2 * m.rows * m.cols;

#pragma omp parallel for
	for(int y = 0; y < m.rows; ++y){
		float *v = m.ptr<float>(y);
		for(int x = 0; x < m.cols; ++x){
			int off = y * m.cols + x;
			dX1[off] = c1 * v[x * m.channels() + 0];
			dX2[off] = c2 * v[x * m.channels() + 1];
			dX3[off] = c3 * v[x * m.channels() + 2];
		}
	}

	res.clipRange(0, 1);

//	QDateTime dt = QDateTime::currentDateTime();
//	QString sdt = dt.toString("yyyy_MM_dd_hh_mm_ss_zzz");
//	std::string stm = sdt.toStdString();

//	cv::Mat s;
//	getMat(res, s, cv::Size(W, W));
//	cv::imwrite(stm + ".jpg", s);
}

void AnnotationReader::getMat(const ct::Matf &in, cv::Mat &out, const cv::Size sz)
{
	if(in.empty())
		return;

	int channels = in.total() / (sz.area());
	if(channels != 3)
		return;

	out = cv::Mat(sz, CV_32FC3);

	float* dX1 = in.ptr() + 0 * out.rows * out.cols;
	float* dX2 = in.ptr() + 1 * out.rows * out.cols;
	float* dX3 = in.ptr() + 2 * out.rows * out.cols;

	for(int y = 0; y < out.rows; ++y){
		float *v = out.ptr<float>(y);
		for(int x = 0; x < out.cols; ++x){
			int off = y * out.cols + x;
			v[x * out.channels() + 0] = dX1[off];
			v[x * out.channels() + 1] = dX2[off];
			v[x * out.channels() + 2] = dX3[off];
		}
	}
	out.convertTo(out, CV_8UC3, 255.);
}
