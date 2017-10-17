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

#include "vocpredict.h"

/////////////////////

void cnv2gpu(std::vector< ct::Matf >& In, std::vector< gpumat::GpuMat >& Out)
{
	Out.resize(In.size());
	for(size_t i = 0; i < In.size(); ++i){
		gpumat::convert_to_gpu(In[i], Out[i]);
	}
}

void cnv2gpu(const std::vector< std::vector< ct::Matf > >& In, std::vector< std::vector< gpumat::GpuMat > >& Out)
{
	Out.resize(In.size());
	for(size_t i = 0; i < In.size(); ++i){
		Out[i].resize(In[i].size());
		for(size_t j = 0; j < In[i].size(); ++j){
			gpumat::convert_to_gpu(In[i][j], Out[i][j]);
		}
	}
}

void cnv2mat(std::vector< std::vector< gpumat::GpuMat > >& In, std::vector< std::vector< ct::Matf > >& Out)
{
	Out.resize(In.size());
	for(size_t i = 0; i < In.size(); ++i){
		Out[i].resize(In[i].size());
		for(size_t j = 0; j < In[i].size(); ++j){
			gpumat::convert_to_mat(In[i][j], Out[i][j]);
		}
	}
}

//////////////////////

VOCGpuTrain::VOCGpuTrain(AnnotationReader *reader)
{
	m_reader = reader;
	if(!m_reader){
		std::cout << "!!!! Annotation Reader not Set. Fail !!!!" << std::endl;
		return;
	}

	m_internal_1 = false;
	m_show_test_image = true;

	m_check_count = 500;

	m_modelSave = "model_voc.bin";

	m_passes = 100000;
	m_batch = 5;
	m_lr = 0.00001;
    m_num_save_pass = 200;

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

void VOCGpuTrain::init()
{
	using namespace meta;

	m_conv.resize(cnv_size2);
	m_mlp.resize(mlp_size);
//	m_mnt_optim.resize(cnv_size);

//	for(size_t i = 0; i < m_conv.size(); ++i){
//		gpumat::convnn_gpu& cnv = m_conv[i];
//		m_mnt_optim[i].setAlpha(m_lr);
//		m_mnt_optim[i].setBetha(0.98);
//		cnv.setOptimizer(&m_mnt_optim[i]);
//	}

	m_conv[0].init(ct::Size(W, H), 3, 3, 64, ct::Size(5, 5), gpumat::LEAKYRELU, false, true, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 2, 64, ct::Size(5, 5), gpumat::LEAKYRELU, false, true, true);
	m_conv[2].init(m_conv[1].szOut(), 64, 1, 128, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[3].init(m_conv[2].szOut(), 128, 1, 256, ct::Size(3, 3), gpumat::LEAKYRELU, true, true, true);
	m_conv[4].init(m_conv[3].szOut(), 256, 2, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[5].init(m_conv[4].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[6].init(m_conv[5].szOut(), 512, 1, 1024, ct::Size(1, 1), gpumat::LEAKYRELU, false, true, true);
	m_conv[7].init(m_conv[6].szOut(), 1024, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);

	m_conv[8].init(m_conv[7].szOut(), 512, 1, 96, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true, true);
//	m_conv[9].init(m_conv[8].szOut(), 1024, 1, 64, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true, true);
//	m_conv[10].init(m_conv[9].szOut(), 1024, 1, Classes + Boxes + Rects, ct::Size(1, 1), gpumat::LINEAR, false, false, true);

	int outf = (Classes + Boxes + Rects) * (K * K);
	m_mlp[0].init(m_conv.back().outputFeatures(), 4096, gpumat::GPU_FLOAT, gpumat::LEAKYRELU);
    m_mlp[1].init(4096,  outf, gpumat::GPU_FLOAT, gpumat::LINEAR);
//	K = m_conv.back().szOut().width;

    m_mlp[0].setDropout(0.9);

	printf("K=%d, conv_out=%d, All_output_features=%d\n", K, m_conv.back().outputFeatures(), outf);

    m_optim_cnv.stop_layer = lrs;

    m_optim_cnv.init(m_conv);
	m_optim_cnv.setAlpha(m_lr);

	m_optim_mlp.init(m_mlp);
	m_optim_mlp.setAlpha(m_lr);
}

void VOCGpuTrain::forward(std::vector<gpumat::GpuMat> &X, std::vector< std::vector< gpumat::GpuMat > > *pY, bool dropout)
{
	if(X.empty() || m_conv.empty())
		return;

	using namespace gpumat;
	using namespace meta;

	std::vector< GpuMat > *pX = &X;

	for(size_t i = 0; i < m_conv.size(); ++i){
		convnn_gpu& cnv = m_conv[i];
		//cnv.setDropout(dropout);
		cnv.forward(pX);
		pX = &cnv.XOut();
	}

    std::vector< GpuMat > *pYm = &m_conv.back().XOut();

    for(size_t i = 0; i < pYm->size(); ++i){
        (*pYm)[i].reshape(1, m_conv.back().outputFeatures());
	}

    m_mlp[0].setDropout(dropout);

	for(size_t i = 0; i < m_mlp.size(); ++i){
        m_mlp[i].forward(pYm);
        pYm = &m_mlp[i].vecA1;
	}

    pYm = &m_mlp.back().vecA1;

	for(size_t i = 0; i < pYm->size(); ++i){
        (*pYm)[i].reshape(K * K, Classes + Rects + Boxes);
	}

	std::vector<int> cols;
	cols.push_back(Classes);
	cols.push_back(Rects);
	cols.push_back(Boxes);

    pY->resize(m_mlp.back().vecA1.size());
	int index = 0;
    for(GpuMat &m: (*pYm)){
		std::vector< GpuMat > &py = (*pY)[index++];
		hsplit2(m, cols, py);
	}

	m_partZ.resize(1);

	for(std::vector< GpuMat > &py: *pY){
		softmax(py[0], 1, m_partZ[0]);
		sigmoid(py[2]);
	}
}

void VOCGpuTrain::backward(std::vector< std::vector< gpumat::GpuMat > > &pY)
{
	using namespace meta;

	m_D.resize(pY.size());
	int index = 0;
	for(std::vector< gpumat::GpuMat > &py: pY){
		gpumat::hconcat2(py, m_D[index++]);
	}

	{
		for(size_t i = 0; i < m_D.size(); ++i){
            m_D[i].reshape(1, (K * K) * (Classes + Rects + Boxes));
		}
//        for(size_t i = 0; i < m_D.size(); ++i){
//            (*(m_mlp.back().pVecA0))[i].reshape(1, (K * K) * (Classes + Rects + Boxes));
//        }

		std::vector< gpumat::GpuMat > *pMlp = &m_D;
		for(int i = m_mlp.size() - 1; i >= 0; --i){
			m_mlp[i].backward(*pMlp);
			pMlp = &m_mlp[i].vecDltA0;
		}
	}

	std::vector< gpumat::GpuMat > *pBack = &m_mlp[0].vecDltA0;

	{
		for(size_t i = 0; i < pBack->size(); ++i){
			(*pBack)[i].reshape((K * K), m_conv.back().kernels);
		}
		std::vector< gpumat::GpuMat > *pCnv = pBack;
		for(int i = m_conv.size() - 1; i >= lrs; --i){
			gpumat::convnn_gpu& cnvl = m_conv[i];
			cnvl.backward(*pCnv, i == lrs);
			pCnv = &cnvl.Dlt;
		}
	}
	m_optim_cnv.pass(m_conv);
	m_optim_mlp.pass(m_mlp);
}

void VOCGpuTrain::predict(std::vector<std::vector<gpumat::GpuMat> > &pY, std::vector<std::vector<Obj> > &res)
{
	std::vector< std::vector< ct::Matf > > mY;
	cnv2mat(pY, mY);
	predict(mY, res);
}

void VOCGpuTrain::predict(std::vector<std::vector< ct::Matf >> &pY, std::vector< std::vector<Obj> > &res)
{
	VocPredict predictor;

	predictor.setReader(m_reader);
	predictor.predict(pY, res);
}

std::vector< std::vector< Obj > > VOCGpuTrain::predicts(std::vector<int> &list, bool show)
{
	std::vector< std::vector< Obj > > res;
	if(!m_reader || list.empty())
		return res;

	using namespace meta;

	std::vector< ct::Matf > mX;
	std::vector< std::vector< ct::Matf > >mY;

	std::vector< gpumat::GpuMat > X;
	std::vector< std::vector< gpumat::GpuMat> > y, t;

	m_reader->getGroundTruthMat(list, Boxes, mX, mY);
	cnv2gpu(mX, X);
	cnv2gpu(mY, y);
//	cnv2gpu(m_reader->lambdaBxs, m_glambdaBxs);

	forward(X, &t);

	if(t.empty())
		return res;

	QDir dir;
	if(!dir.exists("test"))
		dir.mkdir("test");

	for(int k = 0; k < t.size(); ++k){
		{
			gpumat::save_gmat(t[k][0], "test/p_cls" + std::to_string(k));
			gpumat::save_gmat(y[k][0], "test/p_ycls" + std::to_string(k));
		}
		{
			gpumat::save_gmat(t[k][1], "test/p_boxes" + std::to_string(k));
			gpumat::save_gmat(y[k][1], "test/p_ybxs" + std::to_string(k));
		}
		{
			gpumat::save_gmat(t[k][2], "test/p_cfd" + std::to_string(k));
			gpumat::save_gmat(y[k][2], "test/p_ycfd" + std::to_string(k));
		}
	}

	predict(t, res);

	get_result(mX, res, show);

	return res;
}

void VOCGpuTrain::get_result(const std::vector< ct::Matf>& mX, const std::vector<std::vector<Obj> > &res, bool show, int offset)
{
	using namespace meta;

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
			if(!show){
				std::cout << val.name << ": [" << val.p << ", (" << val.rect.x << ", "
						  << val.rect.y << ", " << val.rect.width << ", " << val.rect.height << ")]\n";;
			}

			cv::putText(im, val.name + " p[" + std::to_string(val.p) + "]", val.rect.tl(), 1, 1, cv::Scalar(0, 255, 0), 1);
			cv::rectangle(im, val.rect, cv::Scalar(0, 0, 255), 1);
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

void VOCGpuTrain::predicts(std::string &sdir)
{
	QDir dir(sdir.c_str());
	dir.setNameFilters(QStringList("*.jpg"));

	if(!dir.exists() || m_conv.empty() || !m_reader)
		return;

	std::vector< std::vector< Obj > > res;
	std::vector< gpumat::GpuMat > X;
	std::vector< std::vector< gpumat::GpuMat > >t;
	std::vector< ct::Matf > mX;

	const int max_images = 10;

	mX.resize(max_images);
	for(int i = 0, ind = 0, cnt = 0; i < dir.count(); ++i, ++cnt){
		QString fn = dir.path() + "/" + dir[i];
		ct::Matf &Xi = mX[cnt];
		m_reader->getImage(fn.toStdString(), Xi);

		if(Xi.empty())
			continue;

		mX.push_back(Xi);
		if(cnt >= max_images - 1 || i == dir.count() - 1){
			mX.resize(cnt + 1);
			cnv2gpu(mX, X);

			forward(X, &t);

			res.clear();
			predict(t, res);
			get_result(mX, res, false, ind += mX.size());
			std::cout << "<<<<---- files ended: " << mX.size() << " ------>>>>>>\n";

			mX.clear();
			cnt = -1;
		}
	}
}


void VOCGpuTrain::test_predict()
{
	std::vector<int> list;

	list.push_back(25);
	list.push_back(26);
	list.push_back(125);
	list.push_back(108);
	list.push_back(327);

	predicts(list, true);
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

	m_optim_cnv.setAlpha(lr);
    m_optim_mlp.setAlpha(lr);
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
#if CV_VERSION_MAJOR <= 3 && CV_VERSION_MINOR < 1
	cv::setRNGSeed(seed);
#else
	cv::theRNG().state = seed;
#endif
	ct::generator.seed(seed);
}

void VOCGpuTrain::get_delta(std::vector< std::vector< gpumat::GpuMat >  >& t, std::vector< std::vector< gpumat::GpuMat > >& y, bool test)
{
	using namespace meta;

	for(int b = 0; b < t.size(); ++b){
		std::vector< gpumat::GpuMat >& ti = t[b];
		std::vector< gpumat::GpuMat >& yi = y[b];
		if(test){
			gpumat::save_gmat(m_glambdaBxs[b], "test/lmbd" + std::to_string(b));
			gpumat::save_gmat(ti[0], "test/cls" + std::to_string(b));
			gpumat::save_gmat(yi[0], "test/ycls" + std::to_string(b));
		}
		gpumat::subWithColumn(ti[0], yi[0], m_glambdaBxs[b]);
		if(test){
			gpumat::save_gmat(ti[0], "test/cls_d_" + std::to_string(b));
		}

		if(test){
			gpumat::save_gmat(ti[1], "test/boxes" + std::to_string(b));
			gpumat::save_gmat(yi[1], "test/ybxs" + std::to_string(b));
		}
		gpumat::subWithColumn(ti[1], yi[1], m_glambdaBxs[b]);
		if(test){
			gpumat::save_gmat(ti[1], "test/boxes_d_" + std::to_string(b));
		}

		if(test){
			gpumat::save_gmat(ti[2], "test/cfd" + std::to_string(b));
			gpumat::save_gmat(yi[2], "test/ycfd" + std::to_string(b));
		}
		gpumat::back_delta_sigmoid(ti[2], yi[2], m_glambdaBxs[b]);
		if(test){
			gpumat::save_gmat(ti[2], "test/cfd_d_" + std::to_string(b));
		}
	//		gpumat::sub(t[i], y[i], t[i]);
	}
}

float get_loss(std::vector< std::vector< gpumat::GpuMat > >& t)
{
	using namespace meta;

	ct::Matf mat;
	float res1 = 0, res2 = 0, res3 = 0;
	for(int b = 0; b < t.size(); ++b){
		{
			gpumat::elemwiseSqr(t[b][0], t[b][0]);
			gpumat::convert_to_mat(t[b][0], mat);
			res1 += mat.sum() / mat.rows;
		}
		//res1 /= (last_classes - first_classes + 1);

		{
			gpumat::elemwiseSqr(t[b][1], t[b][1]);
			gpumat::convert_to_mat(t[b][1], mat);
			res2 += mat.sum() / mat.rows;
		}
		//res2 /= (last_boxes - first_boxes + 1);

		{
			gpumat::elemwiseSqr(t[b][2], t[b][2]);
			gpumat::convert_to_mat(t[b][2], mat);
			res3 += mat.sum() / mat.rows;
		}
	}
	//res3 /= (last_confidences - first_confidences + 1);

	return (res1 + res2 + res3) / t.size();
}

void VOCGpuTrain::doPass()
{
	using namespace meta;

	std::vector< ct::Matf > mX;
	std::vector< std::vector< ct::Matf > >mY;

	std::vector< gpumat::GpuMat > X;
	std::vector< std::vector< gpumat::GpuMat > > y, t;
	std::vector< int > list;
	list.resize(m_batch);
	for(int pass = 0; pass < m_passes; ++pass){
		cv::randu(list, 0, m_reader->annotations.size() - 1);
		m_reader->getGroundTruthMat(list, Boxes, mX, mY, true, true);
		cnv2gpu(mX, X);
		cnv2gpu(mY, y);
		cnv2gpu(m_reader->lambdaBxs, m_glambdaBxs);

        forward(X, &t, true);

		get_delta(t, y, (pass % 50) == 0);

		backward(t);

		printf("pass=%d    \r", pass);
		std::cout << std::flush;

		if((pass % m_num_save_pass) == 0 || pass == 30){
			int k = 0;
			float loss = 0;
			int cnt = 0;
			while( k < m_check_count){
				cv::randu(list, 0, m_reader->annotations.size() - 1);
				m_reader->getGroundTruthMat(list, Boxes, mX, mY);
				cnv2gpu(mX, X);
				cnv2gpu(mY, y);
				cnv2gpu(m_reader->lambdaBxs, m_glambdaBxs);

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
        if((pass % 10) == 0 && m_show_test_image){
			test_predict();
		}
		cv::waitKey(1);
	}
}

void VOCGpuTrain::test()
{
	using namespace meta;

	std::vector< ct::Matf > mX;
	std::vector< std::vector< ct::Matf > > mY;
	std::vector< std::vector< Obj > > res;

	std::vector< gpumat::GpuMat > X;
	std::vector< std::vector< gpumat::GpuMat > > y, t;
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

	predict(t, res);
}

void VOCGpuTrain::split_conv(const std::vector<gpumat::GpuMat> &In, std::vector<gpumat::GpuMat> &Out)
{
	int index = 0;
	Out.resize(In.size());
	for(const gpumat::GpuMat& m: In){
		gpumat::transpose(m, Out[index++]);
	}
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

#define USE_MLP 1

	if(m_conv.size() < cnvs)
		m_conv.resize(cnvs);
#if USE_MLP
    if(load_mlp)
        m_mlp.resize(mlps);
#endif
	printf("conv\n");
	for(size_t i = 0; i < cnvs; ++i){
		gpumat::convnn_gpu &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W.rows, cnv.W.cols);
	}

	printf("mlp\n");
	for(size_t i = 0; i < mlps; ++i){
#if USE_MLP
        if(load_mlp){
            gpumat::mlp &mlp = m_mlp[i];
            mlp.read2(fs);
            printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
        }else{
            gpumat::GpuMat W, B;
            gpumat::read_fs2(fs, W);
            gpumat::read_fs2(fs, B);
            printf("layer %d: rows %d, cols %d\n", i, W.rows, W.cols);
        }
#else
		gpumat::GpuMat W, B;
		gpumat::read_fs2(fs, W);
		gpumat::read_fs2(fs, B);
		printf("layer %d: rows %d, cols %d\n", i, W.rows, W.cols);
#endif
	}

	int use_bn = 0, layers = 0;
	fs.read((char*)&use_bn, sizeof(use_bn));
	fs.read((char*)&layers, sizeof(layers));
	if(use_bn > 0){
		for(int i = 0; i < layers; ++i){
			int64_t layer = -1;
			fs.read((char*)&layer, sizeof(layer));
			if(layer >=0 && layer < 10000){
				m_conv[layer].bn.read(fs);
//				gpumat::save_gmat(m_conv[layer].bn.gamma, "g" + std::to_string(layer) +".txt");
//				gpumat::save_gmat(m_conv[layer].bn.betha, "b" + std::to_string(layer) +".txt");
			}
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
		gpumat::convnn_gpu &cnv = m_conv[i];
		cnv.write2(fs);
	}

#if 1
	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write2(fs);
	}
#endif

	int use_bn = 0, layers = 0;
	for(gpumat::convnn_gpu& item: m_conv){
		if(item.use_bn()){
			use_bn = 1;
			layers++;
		}
	}

	fs.write((char*)&use_bn, sizeof(use_bn));
	fs.write((char*)&layers, sizeof(layers));
	if(use_bn > 0){
		for(size_t i = 0; i < m_conv.size(); ++i){
			if(m_conv[i].use_bn()){
				fs.write((char*)&i, sizeof(i));
				m_conv[i].bn.write(fs);
			}
		}
	}

	printf("model saved.\n");
}

void VOCGpuTrain::setModelSaveName(const QString &name)
{
	m_modelSave = name;
}
