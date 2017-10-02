#include "vocpredict.h"
#include <fstream>

#include <QDir>

#include "metaconfig.h"

#ifndef _WIN32
#include <unistd.h>
#endif

VocPredict::VocPredict()
{
	m_lr = 0.00001;
	m_passes = 100000;
	m_batch = 10;
	m_num_save_pass = 100;
	m_check_count = 100;
	m_internal_1 = false;

	m_modelSave = "model_voc.bin";

	m_reader = 0;

	m_out_features = 0;

	using namespace meta;

	{
		m_cols.push_back(Classes);
		m_out_features += Classes;
	}
	{
		m_cols.push_back(Rects);
		m_out_features += (Rects);
	}
	{
		m_cols.push_back(Boxes);
		m_out_features += Boxes;
	}
	printf("Output features: %d, Output matrices: %d\n", m_out_features, m_cols.size());
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

	m_optim_cnv.setAlpha(lr);
}

void VocPredict::init()
{
	using namespace meta;

	m_conv.resize(cnv_size2);

	m_conv[0].init(ct::Size(W, H), 3, 3, 64, ct::Size(5, 5), gpumat::LEAKYRELU, false, true, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 2, 64, ct::Size(5, 5), gpumat::LEAKYRELU, false, true, true);
	m_conv[2].init(m_conv[1].szOut(), 64, 1, 128, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[3].init(m_conv[2].szOut(), 128, 1, 256, ct::Size(3, 3), gpumat::LEAKYRELU, true, true, true);
	m_conv[4].init(m_conv[3].szOut(), 256, 2, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[5].init(m_conv[4].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[6].init(m_conv[5].szOut(), 512, 1, 1024, ct::Size(1, 1), gpumat::LEAKYRELU, false, true, true);
	m_conv[7].init(m_conv[6].szOut(), 1024, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);

	m_conv[8].init(m_conv[7].szOut(), 512, 1, 1024, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true, true);
	m_conv[9].init(m_conv[8].szOut(), 1024, 1, 1024, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true, true);
	m_conv[10].init(m_conv[9].szOut(), 1024, 1, 1024, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true, true);
	m_conv[11].init(m_conv[10].szOut(), 1024, 1, Classes + Boxes + 4 * Boxes, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true, true);

	K = m_conv.back().szOut().width;

	printf("K=%d, All_output_features=%d\n", K, m_conv.back().outputFeatures());

	m_optim_cnv.init(m_conv);
	m_optim_cnv.setAlpha(m_lr);
}

void VocPredict::setReader(AnnotationReader *reader)
{
	m_reader = reader;
}

void VocPredict::forward(std::vector<ct::Matf> &X, std::vector<std::vector<ct::Matf> > *pY)
{
	if(X.empty() || m_conv.empty() || m_mlp.empty())
		return;

	using namespace ct;

	std::vector< Matf > *pX = &X;

	for(size_t i = 0; i < m_conv.size(); ++i){
		conv2::convnn2_mixed& cnv = m_conv[i];
		cnv.forward(pX);
		pX = &cnv.XOut();
	}

	std::vector< Matf > &Y = m_conv.back().XOut();

	std::vector<int> cols;
	cols.push_back(Classes);
	cols.push_back(Rects);
	cols.push_back(Boxes);

	pY->resize(Y.size());
	int index = 0;
	for(Matf &m: Y){
		std::vector< Matf > &py = (*pY)[index++];
		hsplit(m, cols, py);
	}

	for(std::vector< Matf > &py: *pY){
		softmax(py[0], 1);
		sigmoid(py[2]);
	}

}

void VocPredict::backward(std::vector<ct::Matf> &pY)
{
	using namespace meta;
	using namespace ct;

	m_D.resize(pY.size());
	int index = 0;
	for(std::vector< Matf > &py: pY){
		hconcat2(py, m_D[index++]);
	}

	{
		std::vector< ct::Matf > *pCnv = &m_D;
		for(int i = m_conv.size() - 1; i > lrs; --i){
			conv2::convnn2_mixed& cnvl = m_conv[i];
			cnvl.backward(*pCnv, i == lrs);
			pCnv = &cnvl.Dlt;
		}
	}
}

void VocPredict::predict(std::vector<std::vector< ct::Matf > > &pY, std::vector<std::vector<Obj> > &res)
{
	const int Crop = 5;

	using namespace meta;
	using namespace ct;

	int rows = pY.size();

	struct IObj{
		int cls;
		float p;
	};

	float D = W / K;

	res.resize(rows);

	std::vector< ct::Matf > P;
	//std::vector<  IObj > iobj;
	P.resize(pY.size() * Boxes);
	for(int i = 0; i < pY.size(); ++i){
		for(int b = 0; b < Boxes; ++b){
			ct::v_mulColumns(pY[i][0], pY[i][2], P[i * Boxes + b], b);
			ct::v_cropValues<float>(P[i * Boxes + b], 0.1);
		}
	}
	for(int row = 0; row < rows; ++row){
		std::vector< ct::Matf > &py = pY[row];

		for(int b = 0; b < Boxes; ++b){
			Matf Pi = P[row * Boxes + b];
			for(int c = 0; c < Classes; ++c){
				std::vector< float > line;
				for(int j = 0; j < Pi.rows; ++j){
					float *dP = Pi.ptr(j);
					line.push_back(dP[c]);
				}
				crop_sort_classes<float>(line, Crop);
				for(int j = 0; j < Pi.rows; ++j){
					float *dP = Pi.ptr(j);
					dP[c] = line[j];
				}
			}

			for(int j = 0; j < Pi.rows; ++j){
				float *dP = Pi.ptr(j);
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

					int off1 = j;

					int y = off1 / K;
					int x = off1 - y * K;

					ct::Matf& B = py[1];//pY[first_boxes + off1];
					float *dB = B.ptr(j);

					float dx = dB[b * Rects + 0];
					float dy = dB[b * Rects + 1];
					float dw = dB[b * Rects + 2];
					float dh = dB[b * Rects + 3];

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
}

void VocPredict::predicts(std::vector<int> &list, bool show)
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

	get_result(X, res, show);
}

void VocPredict::predicts(std::string &sdir)
{
	QDir dir(sdir.c_str());
	dir.setNameFilters(QStringList("*.jpg"));

	if(!dir.exists() || m_conv.empty() || m_mlp.empty() || !m_reader)
		return;

	std::vector< std::vector< Obj > > res;
	std::vector< ct::Matf > X, t;

	const int max_images = 10;

	X.resize(max_images);
	for(int i = 0, ind = 0, cnt = 0; i < dir.count(); ++i, ++cnt){
		QString fn = dir.path() + "/" + dir[i];
		ct::Matf& Xi = X[cnt];
		Xi.fill(0);
		m_reader->getImage(fn.toStdString(), Xi);

		if(Xi.empty())
			continue;

		if(cnt >= max_images - 1 || i == dir.count() - 1){
			X.resize(cnt + 1);
			forward(X, &t);
			res.clear();
			predict(t, res);
			get_result(X, res, false, ind += X.size());
			std::cout << "<<<<---- files ended: " << X.size() << " ------>>>>>>\n";

			t.clear();
			cnt = -1;
		}
	}
}

void VocPredict::get_result(const std::vector<ct::Matf> &mX, const std::vector<std::vector<Obj> > &res, bool show, int offset)
{
	cv::Mat tmp;
	if(show){
		if(!m_internal_1){
			m_internal_1 = true;
			cv::namedWindow("win", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
		}
	}

	for(size_t i = 0; i < res.size(); ++i){
		const ct::Matf &Xi = mX[i];
		cv::Mat im;
		m_reader->getMat(Xi, im, cv::Size(W, W));

		for(size_t j = 0; j < res[i].size(); ++j){
			const Obj& val = res[i][j];
			if(val.p < 0.5)
				continue;
			if(!show){
				std::cout << val.name << ": [" << val.p << ", (" << val.rect.x << ", "
						  << val.rect.y << ", " << val.rect.width << ", " << val.rect.height << ")]\n";;
			}

			cv::putText(im, val.name + " p[" + std::to_string(val.p) + "]", val.rect.tl(), 1, 1, cv::Scalar(0, 255, 0), 2);
			cv::rectangle(im, val.rect, cv::Scalar(0, 0, 255), 2);
		}

		if(!show){
			cv::imwrite("images/image" + std::to_string(offset + i) + ".jpg", im);
		}else{
			if(tmp.empty())
				im.copyTo(tmp);
			else
				cv::hconcat(tmp, im, tmp);
		}
	}
	if(show){
		cv::imshow("win", tmp);
	}
}

void VocPredict::test_predict()
{
	std::vector<int> list;

	list.push_back(25);
	list.push_back(26);
	list.push_back(125);
	list.push_back(101);
	list.push_back(325);

	predicts(list, true);
}

bool VocPredict::loadModel(const QString &model, bool load_mlp)
{
//	return true;

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

	if((int)m_conv.size() < cnvs)
		m_conv.resize(cnvs);

	printf("conv\n");
	for(int i = 0; i < cnvs; ++i){
		conv2::convnn2_mixed &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W[0].rows, cnv.W[0].cols);
	}

	if(load_mlp){
		m_mlp.resize(mlps);
		printf("mlp\n");
		for(size_t i = 0; i < m_mlp.size(); ++i){
			ct::mlp_mixed &mlp = m_mlp[i];
			mlp.read2(fs);
			printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
		}
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
		conv2::convnn2_mixed &cnv = m_conv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlp_mixed& mlp = m_mlp[i];
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

void save_lambdas(const std::vector<ct::Matf>& lmbd)
{
	std::fstream fs;
	fs.open("test/lmbd.txt", std::ios_base::out);
	if(!fs.is_open())
		return;
	int cnt = lmbd[0].rows;
	for(int r = 0; r < cnt; ++r){
		fs << "<<<<<" << r << ">>>>>" << std::endl;
		for(size_t i = 0; i < lmbd.size(); ++i){
			int y = i / K;
			int x = i % K;
			float *dR = lmbd[i].ptr(r);
			float R = dR[0];
			if(R < 1)
				fs << "---";
			else
				fs << " o ";
			if(x == K - 1)
				fs << std::endl;
		}
		fs << std::endl;
	}
	fs.close();
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
			int cnt = 0;
			while( k < m_check_count){
				cv::randu(list, 0, m_reader->annotations.size() - 1);
				m_reader->getGroundTruthMat(list, Boxes, X, y);




				forward(X, &t);

				get_delta(t, y);

				loss += get_loss(t);

				k += m_batch;
				cnt++;

				printf("test: cur %d, all %d    \r", k, m_check_count);
				std::cout << std::flush;
			}
			loss /= cnt;
			printf("pass=%d, loss=%f    \n", pass, loss);
			saveModel(m_modelSave);
		}
		if((pass % 4) == 0){
			test_predict();
		}
		cv::waitKey(10);
	}

}
