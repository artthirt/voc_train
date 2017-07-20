#include <QCoreApplication>
#include "vocgputrain.h"

#include "custom_types.h"
#include "gpumat.h"
#include "helper_gpu.h"

#include <chrono>

bool contain(const std::map<std::string, std::string>& mp, const std::string& key)
{
	return mp.find(key) != mp.end();
}

std::map<std::string, std::string> parseArgs(int argc, char *argv[])
{
	std::map<std::string, std::string> res;
	for(int i = 0; i < argc; ++i){
		std::string str = argv[i];
		if(str == "-voc" && i < argc){
			res["voc"] = argv[i + 1];
		}
		if(str == "-load_voc" && i < argc){
			res["load_voc"] = argv[i + 1];
		}
		if(str == "-load_pretrain" && i < argc){
			res["load_pretrain"] = argv[i + 1];
		}
		if(str == "-save" && i < argc){
			res["save"] = argv[i + 1];
		}
		if(str == "-image" && i < argc){
			res["image"] = argv[i + 1];
		}
		if(str == "-gpu"){
			res["gpu"] = "1";
		}
		if(str == "-pass" && i < argc){
			res["pass"] = argv[i + 1];
		}
		if(str == "-batch" && i < argc){
			res["batch"] = argv[i + 1];
		}
		if(str == "-lr" && i < argc){
			res["lr"] = argv[i + 1];
		}
		if(str == "-images" && i < argc){
			res["images"] = argv[i + 1];
		}
		if(str == "-seed" && i < argc){
			res["seed"] = argv[i + 1];
		}
		if(str == "-train" && i < argc){
			res["train"] = "1";
		}
	}
	return res;
}

////////////////////

void test(bool save = false)
{
	int rows = 330;
	ct::Matf A(rows, 31), B(rows, 53), C(rows, 84), D(rows, 29);
	gpumat::GpuMat gA, gB, gC, gD, gres;

	std::vector< ct::Matf* > mt;

	mt.push_back(&A);
	mt.push_back(&B);
	mt.push_back(&C);
	mt.push_back(&D);

	for(int i = 0; i < mt.size(); ++i){
		ct::Matf* m = mt[i];

		int id = 0;
		float* dM = m->ptr();
		for(int y = 0; y < m->rows; ++y){
			for(int x = 0; x < m->cols; ++x){
				dM[y * m->cols + x] = x;
			}
		}
	}

	std::vector< gpumat::GpuMat > gmt;
	foreach (ct::Matf* it, mt) {
		gmt.push_back(gpumat::GpuMat());
		gpumat::convert_to_gpu(*it, gmt.back());
	}

	gpumat::hconcat2(gmt, gres);

	if(save){
		gpumat::save_gmat(gres, "tmp.txt");
	}

	std::vector< gpumat::GpuMat > reslist;
	std::vector< int > cols;
	cols.push_back(A.cols);
	cols.push_back(B.cols);
	cols.push_back(C.cols);
	cols.push_back(D.cols);
	gpumat::hsplit2(gres, cols, reslist);

	if(save){
		for(size_t i = 0; i < reslist.size(); ++i){
			QString fn = "temp_" + QString::number(i) + ".txt";
			gpumat::save_gmat(reslist[i], fn.toStdString());
		}
	}

//	std::cout << gres.print() << std::endl;
}

////////////////////

int main(int argc, char *argv[])
{
#if 0
	{
		using namespace std::chrono;

		int64_t cnt = 1000;
		auto start = steady_clock::now();
		for(int i = 0; i < cnt; i++){
			test();
		}
		auto end = steady_clock::now();
		auto elapsed = duration_cast<milliseconds>(end - start);
		long duration = elapsed.count();
		printf("duration %f", (double)duration/cnt);

		test(true);
	}
#endif

	std::map<std::string, std::string> res = parseArgs(argc, argv);

	VOCGpuTrain voc;

	QString voc_dir;

	int passes = 100000, batch = 10;
	float lr = 0.0001;
	bool train = false;

	if(contain(res, "voc")){
		voc_dir = QString::fromStdString(res["voc"]);
	}	
	if(contain(res, "pass")){
		passes = std::stoi(res["pass"]);
	}
	if(contain(res, "batch")){
		batch = std::stoi(res["batch"]);
	}
	if(contain(res, "lr")){
		lr = std::stof(res["lr"]);
	}
	if(contain(res, "seed")){
		int seed = std::stoi(res["seed"]);
		voc.setSeed(seed);
	}
	if(contain(res, "train")){
		train = true;
	}
	bool model_voc_loaded = false;
	if(contain(res, "load_voc")){
		std::string fn = res["load_voc"];
		model_voc_loaded = voc.loadModel(fn.c_str(), true);
		if(model_voc_loaded){
			printf("<<<< model for VOC loaded >>>>\n");
		}
	}
	if(contain(res, "load_pretrain") && !model_voc_loaded){
		std::string fn = res["load_pretrain"];
		voc.loadModel(fn.c_str(), false);
	}
	if(contain(res, "save")){
		std::string fn = res["save"];
		voc.setModelSaveName(fn.c_str());
	}

	if(!voc.setVocFolder(voc_dir)){
		return 1;
	}

	voc.setPasses(passes);
	voc.setBatch(batch);
	voc.setLerningRate(lr);

	int id = 25;
	int cnt = 0;

	if(!train){
		std::vector< ct::Matf > r, im;
		Annotation an = voc.getGroundTruthMat(25, 2, 30, im, r);

		printf("<<<< filename %s >>>>\n", an.filename.c_str());
		for(size_t i = 0; i < an.objs.size(); ++i){
			Obj &obj = an.objs[i];
			printf("obj %s\t x=%d, y=%d, w=%d, h=%d\n", obj.name.c_str(), obj.rects.x, obj.rects.y,
				   obj.rects.width, obj.rects.height);
		}
		for(int i = 0; i < r.size(); ++i){
			std::cout << "-->" << i << std::endl;
			std::cout << r[i].print() << std::endl;
		}
		printf("<<<<<< end >>>>>>>");

		r.resize(0);
		std::vector< int > inds;
		inds.push_back(3);
		inds.push_back(25);
		inds.push_back(21);
		voc.getGroundTruthMat(inds, 2, 30, im, r);
		std::cout << "Mat[0] size: " << r[0].rows << "," << r[0].cols << std::endl;

		for(int i = 0; i < r.size(); ++i){
			std::cout << "-->" << i << std::endl;
			std::cout << r[i].print() << std::endl;
		}

		const int CNT = 100;

		while(1){
			if(!voc.show(id))
				break;

			if(cnt++ > CNT){
				id += 1;
				cnt = 0;
				if(id >= voc.size()){
					id = 0;
				}
			}

			int ch = cv::waitKey(20);
			if(ch == 13)
				break;
		}
	}else{
		voc.doPass();
	}

	return 0;
}
