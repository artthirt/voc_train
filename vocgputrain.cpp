#include "vocgputrain.h"
#include <QDir>
#include <QXmlSimpleReader>
#include <QXmlInputSource>
#include <QMap>

#include <fstream>

#include "gpumat.h"
#include "gpu_mlp.h"
#include "matops.h"

#include "metaconfig.h"

VOCGpuTrain::VOCGpuTrain(AnnotationReader *reader)
{
	m_reader = reader;
	if(!m_reader){
		std::cout << "!!!! Annotation Reader not Set. Fail !!!!" << std::endl;
		return;
	}

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
			for(size_t k = 0; k < P.size(); ++k){
				float *dP = P[k].ptr(i);
				float c = dP[k];
				line.push_back(c);
			}
			crop_sort_classes(line, Crop);
			for(size_t k = 0; k < P.size(); ++k){
				float *dP = P[k].ptr(i);
				dP[k] = line[k];
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
				if(o1.p < dP[k]){
					o1.cls = k;
					o1.p = dP[k];
					printf("%d, %f, %s\n", k, o1.p, get_name(m_reader->classes, o1.cls).c_str());
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

				obj.name = get_name(m_reader->classes, io.cls);
				obj.rect = rec;
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

void VOCGpuTrain::get_delta(std::vector< gpumat::GpuMat >& t, std::vector< gpumat::GpuMat >& y, double lambda, bool test)
{
	for(int i = first_classes, k = 0; i < last_classes + 1; ++i, ++k){
		if(test){
			gpumat::save_gmat(t[i], "test/cls" + std::to_string(k));
			gpumat::save_gmat(y[i], "test/ycls" + std::to_string(k));
		}
		gpumat::subWithColumn(t[i], y[i], m_glambdaBxs[k]);
	}
	for(int i = first_boxes, k = 0; i < last_boxes + 1; ++i, ++k){
		if(test){
			gpumat::save_gmat(t[i], "test/boxes" + std::to_string(k));
			gpumat::save_gmat(y[i], "test/ybxs" + std::to_string(k));
		}
		gpumat::subWithColumn(t[i], y[i], m_glambdaBxs[k]);
	}
	for(int i = first_confidences, k = 0; i < last_confidences + 1; ++i, ++k){
		if(test){
			gpumat::save_gmat(t[i], "test/cfd" + std::to_string(k));
			gpumat::save_gmat(y[i], "test/ycfd" + std::to_string(k));
		}
		gpumat::back_delta_sigmoid(t[i], y[i], m_glambdaBxs[k]);
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
		cv::randu(cols, 0, m_reader->annotations.size() - 1);
		m_reader->getGroundTruthMat(cols, Boxes, mX, mY, true);
		cnv2gpu(mX, X);
		cnv2gpu(mY, y);
		cnv2gpu(m_reader->lambdaBxs, m_glambdaBxs);

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
				cv::randu(cols, 0, m_reader->annotations.size() - 1);
				m_reader->getGroundTruthMat(cols, Boxes, mX, mY);
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
	m_reader->getGroundTruthMat(cols, Boxes, mX, mY, true);
	cnv2gpu(mX, X);
	cnv2gpu(mY, y);

	forward(X, &t);

	predict(t, res, Boxes);
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
