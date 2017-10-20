#ifndef METACONFIG_H
#define METACONFIG_H

namespace meta{
	const int W = 360;
	const int H = W;
	const int Classes = 30;
	const int Boxes = 2;
	const int Rect = 4;
	const int Rects = Rect * Boxes;
	const int K = 9;

	const int cnv_size = 8;
	const int cnv_size2 = 9;
	const int mlp_size = 2;

	const int cnv_do_back_layers = 0;
    const int lrs = cnv_size - 1;
}

#include <vector>
#include <QMap>

/**
 * @brief sort_indexes
 * @param v
 * @return
 */
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
	   [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}

/**
 * @brief crop_sort_classes
 * @param values
 * @param crop
 */
template< typename T >
void crop_sort_classes(std::vector< T >& values, int crop)
{
	std::vector< size_t > idx;
	for(auto a: sort_indexes(values)){
		idx.push_back(a);
	}
	int k = 0;
	for(auto a: idx){
		if(k++ > crop){
			values[a] = 0;
		}
	}
}

/**
 * @brief get_name
 * @param classes
 * @param cls
 * @return
 */
std::string get_name(QMap< std::string, int >& classes, int cls);

#endif // METACONFIG_H
