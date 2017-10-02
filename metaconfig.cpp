#include "metaconfig.h"

namespace meta{
	int K = 7;
}

std::string get_name(QMap<std::string, int> &classes, int cls)
{
	for(QMap< std::string, int >::iterator it = classes.begin(); it != classes.end(); ++it){
		if (it.value() == cls){
			return it.key();
		}
	}
	return "";
}
