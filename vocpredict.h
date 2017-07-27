#ifndef VOCPREDICT_H
#define VOCPREDICT_H

#include "custom_types.h"
#include "convnn2.h"
#include "mlp.h"
#include "matops.h"

#include <QString>

#include "annotationreader.h"

class VocPredict
{
public:
	VocPredict();

	void init();

	void setReader(AnnotationReader* reader);

	void forward(std::vector< ct::Matf >& X, std::vector< ct::Matf >* pY);
	void predict(std::vector< ct::Matf >& pY, std::vector<std::vector<Obj> > &res);
	void predicts(std::vector< int > & list);

	bool loadModel(const QString& model);

private:
	QString m_model;

	std::vector< conv2::convnnf > m_conv;
	std::vector< ct::mlpf > m_mlp;
	ct::Matf m_vec2mat;

	AnnotationReader *m_reader;

	int m_out_features;
	std::vector< int > m_cols;
};

#endif // VOCPREDICT_H
