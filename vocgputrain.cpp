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

void cnv2gpu(const std::vector< ct::Matf >& In, std::vector< gpumat::GpuMat >& Out)
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
	m_num_save_pass = 300;

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
	m_conv.resize(cnv_size2);
//	m_mnt_optim.resize(cnv_size);

//	for(size_t i = 0; i < m_conv.size(); ++i){
//		gpumat::convnn_gpu& cnv = m_conv[i];
//		m_mnt_optim[i].setAlpha(m_lr);
//		m_mnt_optim[i].setBetha(0.98);
//		cnv.setOptimizer(&m_mnt_optim[i]);
//	}
	using namespace meta;

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

void VOCGpuTrain::forward(std::vector<gpumat::GpuMat> &X, std::vector< std::vector< gpumat::GpuMat > > *pY, bool dropout)
{
	if(X.empty() || m_conv.empty() || m_mlp.empty())
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

	std::vector< GpuMat > &Y = m_conv.back().XOut();

	std::vector<int> cols;
	cols.push_back(Classes);
	cols.push_back(Rects);
	cols.push_back(Boxes);

	pY->resize(Y.size());
	int index = 0;
	for(GpuMat &m: Y){
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
		std::vector< gpumat::GpuMat > *pCnv = &m_D;
		for(int i = m_conv.size() - 1; i >= lrs; --i){
			gpumat::convnn_gpu& cnvl = m_conv[i];
			cnvl.backward(*pCnv, i == lrs);
			pCnv = &cnvl.Dlt;
		}
	}
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

	forward(X, &t);

	if(t.empty())
		return res;

	QDir dir;
	if(!dir.exists("test"))
		dir.mkdir("test");

	for(int i = first_classes, k = 0; i < last_classes + 1; ++i, ++k){
		gpumat::save_gmat(t[i], "test/cls" + std::to_string(k));
		gpumat::save_gmat(y[i], "test/ycls" + std::to_string(k));
	}
	for(int i = first_boxes, k = 0; i < last_boxes + 1; ++i, ++k){
		gpumat::save_gmat(t[i], "test/boxes" + std::to_string(k));
		gpumat::save_gmat(y[i], "test/ybxs" + std::to_string(k));
	}
	for(int i = first_confidences, k = 0; i < last_confidences + 1; ++i, ++k){
		gpumat::save_gmat(t[i], "test/cfd" + std::to_string(k));
		gpumat::save_gmat(y[i], "test/ycfd" + std::to_string(k));
	}

	predict(t, res);

	get_result(mX, res, show);

	return res;
}

void VOCGpuTrain::get_result(const std::vector< ct::Matf>& mX, const std::vector<std::vector<Obj> > &res, bool show, int offset)
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

void VOCGpuTrain::predicts(std::string &sdir)
{
	QDir dir(sdir.c_str());
	dir.setNameFilters(QStringList("*.jpg"));

	if(!dir.exists() || m_conv.empty() || m_mlp.empty() || !m_reader)
		return;

	std::vector< std::vector< Obj > > res;
	std::vector< gpumat::GpuMat > X, t;
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
	list.push_back(101);
	list.push_back(325);

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

void VOCGpuTrain::get_delta(std::vector< std::vector< gpumat::GpuMat >  >& t, std::vector< std::vector< gpumat::GpuMat > >& y, bool test)
{
	using namespace meta;

	for(int b = 0; b < t.size(); ++b){
		std::vector< gpumat::GpuMat >& ti = t[b];
		std::vector< gpumat::GpuMat >& yi = y[b];
		if(test){
			gpumat::save_gmat(ti[0], "test/cls" + std::to_string(b));
			gpumat::save_gmat(yi[0], "test/ycls" + std::to_string(b));
		}
		gpumat::subWithColumn(ti[0], yi[0], m_glambdaBxs[b]);

		if(test){
			gpumat::save_gmat(ti[1], "test/boxes" + std::to_string(b));
			gpumat::save_gmat(yi[1], "test/ybxs" + std::to_string(b));
		}
		gpumat::subWithColumn(ti[1], yi[1], m_glambdaBxs[b]);

		if(test){
			gpumat::save_gmat(ti[2], "test/cfd" + std::to_string(b));
			gpumat::save_gmat(yi[2], "test/ycfd" + std::to_string(b));
		}
		gpumat::back_delta_sigmoid(ti[2], yi[2], m_glambdaBxs[b]);
	//		gpumat::sub(t[i], y[i], t[i]);
	}
}

float get_loss(std::vector< gpumat::GpuMat >& t)
{
	ct::Matf mat;
	float res1 = 0, res2 = 0, res3 = 0;
	for(int b = 0; b < t.size(); ++b){
		{
			gpumat::elemwiseSqr(t[b][0], t[b][0]);
			gpumat::convert_to_mat(t[b], mat);
			res1 = mat.sum() / mat.rows;
		}
		//res1 /= (last_classes - first_classes + 1);

		{
			gpumat::elemwiseSqr(t[i], t[i]);
			gpumat::convert_to_mat(t[i], mat);
			res2 = mat.sum() / mat.rows;
		}
		//res2 /= (last_boxes - first_boxes + 1);

		{
			gpumat::elemwiseSqr(t[i], t[i]);
			gpumat::convert_to_mat(t[i], mat);
			res3 = mat.sum() / mat.rows;
		}
	}
	//res3 /= (last_confidences - first_confidences + 1);

	return res1 + res2 + res3;
}

void VOCGpuTrain::doPass()
{
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

	if((int)m_conv.size() < cnvs)
		m_conv.resize(cnvs);

	printf("conv\n");
	for(int i = 0; i < cnvs; ++i){
		gpumat::convnn_gpu &cnv = m_conv[i];
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
		gpumat::convnn_gpu &cnv = m_conv[i];
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
