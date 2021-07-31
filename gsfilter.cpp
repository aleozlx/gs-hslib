#include "heterogeneous_superpixel.hpp"
#include "heterogeneous_superpixel_optional.hpp"

using namespace heterogeneous_superpixel;
using namespace heterogeneous_superpixel::optional;

struct RawFrameMetadata {
	unsigned int width, height;
};

int main(int argc, const char* argv[]) {
	RawFrameMetadata metadata{1920, 1080};

	return 0;
}
