#ifndef VOCGPUTRAIN_H
#define VOCGPUTRAIN_H

#include <QString>
#include <QXmlDefaultHandler>
#include <QMap>
#include <vector>

#include <opencv2/opencv.hpp>

#include "custom_types.h"
#include "gpumat.h"
#include "gpu_mlp.h"
#include "convnn2_gpu.h"

#include "annotationreader.h"

class VOCGpuTrain
{
public:
	VOCGpuTrain(AnnotationReader* reader);

	bool loadModel(const QString& model, bool load_mlp = false);
	void saveModel(const QString& name);
	void setModelSaveName(const QString& name);

	void setTestReader(AnnotationReader* reader);

	void init();

	void forward(std::vector< gpumat::GpuMat >& X, std::vector<std::vector<gpumat::GpuMat> > *pY, bool dropout = false);
	void backward(std::vector<std::vector<gpumat::GpuMat> > &pY);

	void predict(std::vector< std::vector< gpumat::GpuMat> >& pY, std::vector< std::vector<Obj> >& res);
	void predict(std::vector<std::vector<ct::Matf> > &pY, std::vector<std::vector<Obj> > &res);
	std::vector<std::vector<Obj> > predicts(std::vector< int > & list, bool show = false);
	void predicts(std::string& sdir);

	void get_result(const std::vector<ct::Matf> &mX, const std::vector< std::vector< Obj > >& res, bool show, int offset = 0);

	void test_predict();

	int passes() const;
	void setPasses(int passes);
	int batch() const;
	void setBatch(int batch);
	float lr() const;
	void setLerningRate(float lr);
	int numSavePass() const;
	void setNumSavePass(int num);
	void setSeed(int seed);

	void doPass();

	void test();

private:
	AnnotationReader* m_reader;
	AnnotationReader* m_reader_test;

	QString m_model;

	std::vector< gpumat::GpuMat > m_glambdaBxs;

	//////////

	int m_passes;
	int m_batch;
	float m_lr;
	int m_num_save_pass;
	int m_check_count;

	QString m_modelSave;

	bool m_internal_1;
	bool m_show_test_image;

	std::vector< gpumat::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;
	gpumat::GpuMat m_vec2mat;
//	gpumat::GpuMat m_D;
	gpumat::GpuMat m_T;
	gpumat::CnvMomentumOptimizer m_optim_cnv;
	gpumat::MlpOptimMoment m_optim_mlp;
	std::vector< int > m_cols;
	int m_out_features;

	std::vector< gpumat::GpuMat > m_D;
	std::vector< gpumat::GpuMat > m_partZ;
	std::vector< gpumat::GpuMat > m_delta_cnv;

	//////////

	void split_conv(const std::vector< gpumat::GpuMat > &In, std::vector< gpumat::GpuMat > &Out);

	void get_delta(std::vector<std::vector<gpumat::GpuMat> > &t, std::vector<std::vector<gpumat::GpuMat> > &y, bool test = false);
};

#endif // VOCGPUTRAIN_H
