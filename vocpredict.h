#ifndef VOCPREDICT_H
#define VOCPREDICT_H

#include "custom_types.h"
#include "convnn2.h"
#include "convnn2_mixed.h"
#include "mlp_mixed.h"
#include "mlp.h"
#include "matops.h"

#include <QString>

#include "annotationreader.h"

class VocPredict
{
public:
	VocPredict();

	void setPasses(int val);
	void setBatch(int val);
	void setLr(float lr);

	void init();

	void setReader(AnnotationReader* reader);

	void forward(std::vector< ct::Matf >& X, std::vector< std::vector< ct::Matf> >* pY);
	void backward(std::vector<std::vector<ct::Matf> > &pY);
	void predict(std::vector<std::vector<ct::Matf> > &pY, std::vector<std::vector<Obj> > &res);
	void predicts(std::vector< int > & list, bool show = false);
	void predicts(std::string& sdir);

	void get_result(const std::vector<ct::Matf> &mX, const std::vector< std::vector< Obj > >& res, bool show, int offset = 0);

	void test_predict();

	bool loadModel(const QString& model, bool load_mlp = true);
	void saveModel(const QString &name);
	void setModelSaveName(const QString& name);
	void setSeed(int seed);

	void doPass();

	void get_delta(std::vector<std::vector<ct::Matf> > &t, std::vector<std::vector<ct::Matf> > &y, bool test = false);

private:
	QString m_model;
	int m_passes;
	int m_batch;
	float m_lr;
	int m_num_save_pass;
	int m_check_count;
	QString m_modelSave;
	bool m_internal_1;

	std::vector< conv2::convnn2_mixed > m_conv;
	std::vector< ct::mlp_mixed > m_mlp;
	std::vector< ct::Matf> m_D;
	std::vector< ct::Matf > m_delta_cnv;

	AnnotationReader *m_reader;

	conv2::CnvMomentumOptimizerMixed m_optim_cnv;
	ct::MlpMomentumOptimizerMixed m_optim_mlp;

	int m_out_features;
	std::vector< int > m_cols;
};

#endif // VOCPREDICT_H
