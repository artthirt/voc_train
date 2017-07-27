#ifndef ANNOTATIONREADER_H
#define ANNOTATIONREADER_H

#include <QString>
#include <QXmlDefaultHandler>

#include "custom_types.h"

#include <opencv2/opencv.hpp>

#include <QMap>

struct Obj{
	Obj(){
		p = 0;
	}

	std::string name;
	cv::Rect rect;
	cv::Rect2f rectf;
	float p;
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
		return std::min(xmin, xmax);
	}
	int y() const{
		return std::min(ymin, ymax);
	}
	int w() const{
		return std::abs(xmax - xmin);
	}
	int h() const{
		return std::abs(ymax - ymin);
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

class AnnotationReader: public QXmlDefaultHandler
{
public:
	AnnotationReader();
	bool setVocFolder(const QString sdir);

	/// array of annotations
	std::vector< Annotation > annotations;
	QMap< std::string, int > classes;
	std::vector< ct::Matf > lambdaBxs;

	size_t size() const;


	Annotation &getGroundTruthMat(int index, int boxes, std::vector<ct::Matf> &images,
						   std::vector< ct::Matf >& res, int row = 0, int rows = 1, bool flip = false, bool load_image = true);
	void getGroundTruthMat(std::vector<int> indices, int boxes,
						   std::vector<ct::Matf> &images, std::vector< ct::Matf >& res, bool flip = false);

	void getImage(const std::string& filename, ct::Matf& res, bool flip = false);

	void getMat(const ct::Matf& in, cv::Mat& out, const cv::Size sz);

	bool show(int index, bool flip = false, const std::string name = "out");

private:
	QString m_vocdir;
	QString m_key;
	QString m_value;
	Annotation* m_currentAn;
	Obj* m_curObj;
	BndBox m_box;
	cv::Size* m_refSize;
	cv::Mat m_sample;
	int m_index;

	bool load_annotation(const QString& fileName, Annotation& annotation);
	void update_output(std::vector< ct::Matf >& res, Obj& ob, int off, int bxid, int row);

	// QXmlContentHandler interface
public:
	bool startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &atts);
	bool endElement(const QString &namespaceURI, const QString &localName, const QString &qName);
	bool characters(const QString &ch);
	QString errorString() const;

};

#endif // ANNOTATIONREADER_H
