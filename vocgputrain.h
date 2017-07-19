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

	bool show(int index);

	size_t size() const;

	void loadModel(const QString& model);

	void getGroundTruthMat(int index, int boxes, int classes, std::vector<ct::Matf> &images,
						   std::vector< ct::Matf >& res, int row = 0, int rows = 1);
	void getGroundTruthMat(std::vector<int> indices, int boxes, int classes,
						   std::vector<ct::Matf> &images, std::vector< ct::Matf >& res);

	void getImage(const std::string& filename, ct::Matf& res, bool flip = false);

	void init();

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

	std::vector< gpumat::conv2::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;

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
