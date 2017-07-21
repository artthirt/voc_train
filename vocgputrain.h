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

struct Obj{
	std::string name;
	cv::Rect rects;
};

struct BndBox{
	BndBox(){
		xmin = ymin = xmax = ymax = 0;
	}
	bool empty(){
		return xmin == 0 && xmax == 0 && ymin == 0&& ymax == 0;
	}
	void clear(){
		xmin = ymin = xmax = ymax = 0;
	}
	int x() const{
		return xmin;
	}
	int y() const{
		return ymin;
	}
	int w() const{
		return xmax - xmin;
	}
	int h() const{
		return ymax - ymin;
	}

	int xmin;
	int ymin;
	int xmax;
	int ymax;
};

struct Annotation{
	std::string folder;
	std::string filename;
	std::vector< Obj > objs;
	cv::Size size;
};

class VOCGpuTrain: public QXmlDefaultHandler
{
public:
	VOCGpuTrain();
	bool setVocFolder(const QString sdir);

	bool show(int index, bool flip = false, const std::__cxx11::string name = "out");

	size_t size() const;

	bool loadModel(const QString& model, bool load_mlp = false);
	void saveModel(const QString& name);
	void setModelSaveName(const QString& name);

	Annotation &getGroundTruthMat(int index, int boxes, std::vector<ct::Matf> &images,
						   std::vector< ct::Matf >& res, int row = 0, int rows = 1, bool flip = false);
	void getGroundTruthMat(std::vector<int> indices, int boxes,
						   std::vector<ct::Matf> &images, std::vector< ct::Matf >& res, bool flip = false);

	void getImage(const std::string& filename, ct::Matf& res, bool flip = false);

	void init();

	void forward(std::vector< gpumat::GpuMat >& X, std::vector< gpumat::GpuMat >* pY);
	void backward(std::vector< gpumat::GpuMat >& pY);

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

private:
	QString m_vocdir;
	QString m_model;
	std::vector< Annotation > m_annotations;
	QString m_key;
	QString m_value;
	Annotation* m_currentAn;
	Obj* m_curObj;
	BndBox m_box;
	cv::Size* m_refSize;
	cv::Mat m_sample;
	int m_index;

	//////////

	int m_passes;
	int m_batch;
	float m_lr;
	int m_num_save_pass;
	int m_check_count;

	QString m_modelSave;

	std::vector< gpumat::conv2::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;
	gpumat::GpuMat m_vec2mat;
	gpumat::GpuMat m_D;
	gpumat::MlpOptim m_optim;
	std::vector< int > m_cols;
	int m_out_features;

	std::vector< gpumat::GpuMat > m_partZ;
	std::vector< gpumat::GpuMat > m_delta_cnv;

	//////////

	QMap< std::string, int > m_classes;

	bool load_annotation(const QString& fileName, Annotation& annotation);

	// QXmlContentHandler interface
public:
	bool startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &atts);
	bool endElement(const QString &namespaceURI, const QString &localName, const QString &qName);
	bool characters(const QString &ch);
	QString errorString() const;
};

#endif // VOCGPUTRAIN_H
