#include "vocpredict.h"
#include <fstream>

#include <QDir>

#include "metaconfig.h"

VocPredict::VocPredict()
{
	m_lr = 0.00001;
	m_passes = 100000;
	m_batch = 10;
	m_num_save_pass = 30;
	m_check_count = 100;

	m_modelSave = "model_voc.bin";

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

void VocPredict::setPasses(int val)
{
	m_passes = val;
}

void VocPredict::setBatch(int val)
{
	m_batch = val;
}

void VocPredict::setLr(float lr)
{
	m_lr = lr;

	m_optim.setAlpha(lr);
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

	m_optim.init(m_mlp);
	m_optim.setAlpha(m_lr);
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

void VocPredict::backward(std::vector<ct::Matf> &pY)
{
	ct::hconcat2(pY, m_D);

	ct::Matf *pD = &m_D;
	for(int i = m_mlp.size() - 1; i > -1; i--){
		ct::mlpf& mlp = m_mlp[i];
		mlp.backward(*pD, i == 0 && cnv_do_back_layers == 0);
		pD = &mlp.DltA0;
	}
	m_optim.pass(m_mlp);

	if(cnv_do_back_layers > 0){
		ct::mlpf& mlp0 = m_mlp.front();
		conv2::convnnf& cnvl = m_conv.back();
		conv2::mat2vec(mlp0.DltA0, cnvl.szK, m_delta_cnv);
		std::vector< ct::Matf > *pCnv = &m_delta_cnv;
		for(int i = m_conv.size() - 1; i > lrs; --i){
			conv2::convnnf& cnvl = m_conv[i];
			pCnv = &cnvl.Dlt;
		}
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

	float D = W / K;

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
	for(int row = 0; row < rows; ++row){
		IObj iobj[K * K * Boxes];

		for(int c = 0; c < Classes; ++c){
			std::vector< float > line;
			for(int j = 0; j < P.size(); ++j){
				float *dP = P[j].ptr(row);
				line.push_back(dP[c]);
			}
			sort_indexes<float>(line);
			crop_sort_classes<float>(line, Crop);
			for(int j = 0; j < P.size(); ++j){
				float *dP = P[j].ptr(row);
				dP[c] = line[j];
			}
		}

		for(int j = 0; j < P.size(); ++j){
			float *dP = P[j].ptr(row);
			float p = dP[0]; int id = 0;
			for(int c = 1; c < Classes; ++c){
				if(p < dP[c]){
					p = dP[c];
					id = c;
				}
			}
			if(p > 0.1 && id > 0){
				Obj ob;
				ob.p = p;
				ob.name = get_name(m_reader->classes, id);

				int off2 = j % Boxes;
				int off1 = (j - off2)/Boxes;


				int y = off1 / K;
				int x = off1 - y * K;

				ct::Matf& B = pY[first_boxes + off1];
				float *dB = B.ptr(row);

				float dx = dB[off2 * 4 + 0];
				float dy = dB[off2 * 4 + 1];
				float dw = dB[off2 * 4 + 2];
				float dh = dB[off2 * 4 + 3];

				float w = dw * dw * W;
				float h = dh * dh * W;

				if(!w || !h)
					continue;

				ob.rectf = cv::Rect2f(dx, dy, dw, dh);
				ob.rect = cv::Rect((float)x * D + dx * D - w/2,
								   (float)y * D + dy * D - h/2,
								   w, h);
				res[row].push_back(ob);
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

	m_reader->getGroundTruthMat(list, Boxes, X, y, true, true);

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
		ct::Matf &Xi = X[i];
		cv::Mat im;
		m_reader->getMat(Xi, im, cv::Size(W, W));

		for(size_t j = 0; j < res[i].size(); ++j){
			Obj& val = res[i][j];
			std::cout << val.name << ": [" << val.p << ", (" << val.rect.x << ", "
					  << val.rect.y << ", " << val.rect.width << ", " << val.rect.height << ")]\n";;

			cv::putText(im, val.name, val.rect.tl(), 1, 1, cv::Scalar(0, 0, 255), 2);
			cv::rectangle(im, val.rect, cv::Scalar(0, 255, 55), 2);
		}
		cv::imwrite("test/image" + std::to_string(i) + ".jpg", im);
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

void VocPredict::saveModel(const QString &name)
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
		conv2::convnnf &cnv = m_conv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlpf& mlp = m_mlp[i];
		mlp.write2(fs);
	}

	printf("model saved.\n");
}

void VocPredict::setModelSaveName(const QString &name)
{
	m_modelSave = name;
}

void VocPredict::setSeed(int seed)
{
	cv::setRNGSeed(seed);
}

void VocPredict::get_delta(std::vector< ct::Matf >& t, std::vector< ct::Matf >& y, bool test)
{
	for(int i = first_classes, k = 0; i < last_classes + 1; ++i, ++k){
		if(test){
			ct::save_mat(t[i], "test/cls" + std::to_string(k));
			ct::save_mat(y[i], "test/ycls" + std::to_string(k));
		}
		ct::subWithColumn(t[i], y[i], m_reader->lambdaBxs[k]);
	}
	for(int i = first_boxes, k = 0; i < last_boxes + 1; ++i, ++k){
		if(test){
			ct::save_mat(t[i], "test/boxes" + std::to_string(k));
			ct::save_mat(y[i], "test/ybxs" + std::to_string(k));
		}
		ct::subWithColumn(t[i], y[i], m_reader->lambdaBxs[k]);
	}
	for(int i = first_confidences, k = 0; i < last_confidences + 1; ++i, ++k){
		if(test){
			ct::save_mat(t[i], "test/cfd" + std::to_string(k));
			ct::save_mat(y[i], "test/ycfd" + std::to_string(k));
		}
		ct::back_delta_sigmoid(t[i], y[i], m_reader->lambdaBxs[k]);
//		gpumat::sub(t[i], y[i], t[i]);
	}
}

float get_loss(std::vector< ct::Matf >& t)
{
	float res1 = 0;
	for(int i = first_classes; i < last_classes + 1; ++i){
		ct::v_elemwiseSqr(t[i]);
		res1 += t[i].sum() / t[i].rows;
	}
	res1 /= (last_classes - first_classes + 1);

	float res2 = 0;
	for(int i = first_boxes; i < last_boxes + 1; ++i){
		ct::v_elemwiseSqr(t[i]);
		res2 += t[i].sum() / t[i].rows;
	}
	res2 /= (last_boxes - first_boxes + 1);

	float res3 = 0;
	for(int i = first_confidences; i < last_confidences + 1; ++i){
		ct::v_elemwiseSqr(t[i]);
		res3 += t[i].sum() / t[i].rows;
	}
	res3 /= (last_confidences - first_confidences + 1);

	return res1 + res2 + res3;
}

void VocPredict::doPass()
{
	if(!m_reader)
		return;

	std::vector< ct::Matf > X, y, t;
	std::vector< int > list;
	list.resize(m_batch);

	for(int pass = 0; pass < m_passes; ++pass){

		cv::randu(list, 0, m_reader->annotations.size() - 1);

		m_reader->getGroundTruthMat(list, Boxes, X, y, true, true);

		forward(X, &t);

		get_delta(t, y, (pass % 100) == 0);

		backward(t);

		printf("pass=%d    \r", pass);
		std::cout << std::flush;

		if((pass % m_num_save_pass) == 0 && pass > 0 || pass == 30){
			int k = 0;
			float loss = 0;
			while( k < m_check_count){
				cv::randu(list, 0, m_reader->annotations.size() - 1);
				m_reader->getGroundTruthMat(list, Boxes, X, y);

				forward(X, &t);

				get_delta(t, y);

				loss += get_loss(t);

				k += m_batch;

				printf("test: cur %d, all %d    \r", k, m_check_count);
				std::cout << std::flush;
			}
			loss /= m_check_count;
			printf("pass=%d, loss=%f    \n", pass, loss);
			saveModel(m_modelSave);
		}
	}

}