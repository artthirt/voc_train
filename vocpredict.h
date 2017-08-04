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

	void forward(std::vector< ct::Matf >& X, std::vector< ct::Matf >* pY);
	void backward(std::vector< ct::Matf >& pY);
	void predict(std::vector< ct::Matf >& pY, std::vector<std::vector<Obj> > &res);
	void predicts(std::vector< int > & list);

	bool loadModel(const QString& model);
	void saveModel(const QString &name);
	void setModelSaveName(const QString& name);
	void setSeed(int seed);

	void doPass();

	void get_delta(std::vector< ct::Matf >& t, std::vector< ct::Matf >& y, bool test = false);

private:
	QString m_model;
	int m_passes;
	int m_batch;
	float m_lr;
	int m_num_save_pass;
	int m_check_count;
	QString m_modelSave;

	ct::MlpOptim_mixed m_optim;

	std::vector< conv2::convnn2_mixed > m_conv;
	std::vector< ct::mlp_mixed > m_mlp;
	ct::Matf m_vec2mat;
	ct::Matf m_D;
	std::vector< ct::Matf > m_delta_cnv;

	AnnotationReader *m_reader;

	int m_out_features;
	std::vector< int > m_cols;
};

#endif // VOCPREDICT_H
