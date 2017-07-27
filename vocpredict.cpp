#include "vocpredict.h"
#include <fstream>

#include <QDir>

#include "metaconfig.h"

VocPredict::VocPredict()
{
	m_reader = 0;

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
}

void VocPredict::init()
{
	m_conv.resize(cnv_size);

	m_conv[0].init(ct::Size(W, W), 3, 4, 64, ct::Size(7, 7), true, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 1, 256, ct::Size(5, 5), true);
	m_conv[2].init(m_conv[1].szOut(), 256, 1, 512, ct::Size(3, 3), true);
	m_conv[3].init(m_conv[2].szOut(), 512, 1, 1024, ct::Size(3, 3), true);
//	m_conv[4].init(m_conv[3].szOut(), 1024, 1, 1024, ct::Size(3, 3), false);

	int outFeatures = m_conv.back().outputFeatures();

	m_mlp.resize(mlp_size);

	m_mlp[0].init(outFeatures, 4096);
	m_mlp[1].init(4096, 2048);
	m_mlp[2].init(2048, m_out_features);
}

void VocPredict::setReader(AnnotationReader *reader)
{
	m_reader = reader;
}

void VocPredict::forward(std::vector<ct::Matf> &X, std::vector<ct::Matf> *pY)
{
	if(X.empty() || m_conv.empty() || m_mlp.empty())
		return;

	using namespace ct;

	std::vector< Matf > *pX = &X;

	for(size_t i = 0; i < m_conv.size(); ++i){
		conv2::convnnf& cnv = m_conv[i];
		cnv.forward(pX, RELU);
		pX = &cnv.XOut();
	}

	conv2::vec2mat(m_conv.back().XOut(), m_vec2mat);

	Matf *pX2 = &m_vec2mat;

	etypefunction func = RELU;

	for(size_t i = 0; i < m_mlp.size(); ++i){
		if(i == m_mlp.size() - 1)
			func = LINEAR;
		mlpf& _mlp = m_mlp[i];
		_mlp.forward(pX2, func);
		pX2 = &_mlp.Y();
	}

	hsplit(*pX2, m_cols, *pY);

	for(int i = first_classes; i < last_classes + 1; ++i){
		(*pY)[i] = softmax((*pY)[i], 1);
	}
	for(int i = first_confidences; i < last_confidences + 1; ++i){
		v_sigmoid((*pY)[i]);
	}
}

void VocPredict::predict(std::vector<ct::Matf> &pY, std::vector<std::vector<Obj> > &res)
{
	const int Crop = 5;

	int rows = pY[0].rows;

	struct IObj{
		int cls;
		float p;
	};

	res.resize(rows);

	std::vector< ct::Matf > P;
	//std::vector<  IObj > iobj;
	P.resize(K * K * Boxes);
	for(int i = 0; i < K * K; ++i){
		for(int b = 0; b < Boxes; ++b){
			ct::v_mulColumns(pY[first_classes + i], pY[first_confidences + i], P[i * Boxes + b], b);
			ct::v_cropValues<float>(P[i * Boxes + b], 0.1);
		}
	}
	for(int i = 0; i < rows; ++i){
		IObj iobj[K * K * Boxes];
		for(int j = 0; j < Classes; ++j){
			std::vector<float> line;
			for(size_t k = first_classes; k < last_classes + 1; ++k){
				float *dP = pY[k].ptr(i);
				float c = dP[j];
				line.push_back(c);
			}
			crop_sort_classes(line, Crop);
			for(size_t k = 0; k < P.size(); ++k){
				float *dP = P[k].ptr(i);
				dP[j] = line[j];
			}
		}
		for(size_t j = 0; j < P.size(); ++j){
			ct::Matf& Pj = P[j];
			float *dP = Pj.ptr(i);

			IObj o1;
			o1.cls = 0;
			o1.p = dP[0];
			bool f = false;
			for(int k = 1; k < Classes; ++k){
				if(o1.p < dP[k] && dP[k] > 0.1){
					o1.cls = k;
					o1.p = dP[k];
					printf("%d, %f, %s\n", k, o1.p, get_name(m_reader->classes, o1.cls).c_str());
					f = true;

				}
			}
			if(f)
				iobj[j] = o1;
		}

		for(int k = 0; k < P.size(); ++k){
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

				float dx = dB[off2 * 4 + 0];
				float dy = dB[off2 * 4 + 1];
				float dw = dB[off2 * 4 + 2];
				float dh = dB[off2 * 4 + 3];

				if(dx < 0 || dx > 1 || dy < 0 || dy > 1 || dw < 0 || dh < 0)
					continue;

				rec.x = offx * D + dx * D;
				rec.y = offy * D + dy * D;
				rec.width = dw * W;
				rec.height = dh * W;

				obj.name = get_name(m_reader->classes, io.cls);
				obj.rect = rec;
				obj.p = io.p;
				res[i].push_back(obj);
			}
		}

	}
}

void VocPredict::predicts(std::vector<int> &list)
{
	if(!m_reader || list.empty())
		return;

	std::vector< ct::Matf > X, y, t;
	std::vector< std::vector< Obj > > res;

	m_reader->getGroundTruthMat(list, Boxes, X, y);

	forward(X, &t);

	if(t.empty())
		return;

	QDir dir;
	if(!dir.exists("test"))
		dir.mkdir("test");

	for(int i = first_classes, k = 0; i < last_classes + 1; ++i, ++k){
		ct::save_mat(t[i], "test/cls" + std::to_string(k));
		ct::save_mat(y[i], "test/ycls" + std::to_string(k));
	}
	for(int i = first_boxes, k = 0; i < last_boxes + 1; ++i, ++k){
		ct::save_mat(t[i], "test/boxes" + std::to_string(k));
		ct::save_mat(y[i], "test/ybxs" + std::to_string(k));
	}
	for(int i = first_confidences, k = 0; i < last_confidences + 1; ++i, ++k){
		ct::save_mat(t[i], "test/cfd" + std::to_string(k));
		ct::save_mat(y[i], "test/ycfd" + std::to_string(k));
	}

	predict(t, res);

	for(size_t i = 0; i < res.size(); ++i){
		for(size_t j = 0; j < res[i].size(); ++j){
			Obj& val = res[i][j];
			std::cout << val.name << ": [" << val.p << ", (" << val.rect.x << ", "
					  << val.rect.y << ", " << val.rect.width << ", " << val.rect.height << ")]\n";;

		}
	}
}

bool VocPredict::loadModel(const QString &model)
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
		conv2::convnnf &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W[0].rows, cnv.W[0].cols);
	}

	m_mlp.resize(mlps);
	printf("mlp\n");
	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlpf &mlp = m_mlp[i];
		mlp.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
	}

	printf("model loaded.\n");
	return true;
}
