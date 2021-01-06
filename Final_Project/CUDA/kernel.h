# ifndef __KERNEL_H__
# define __KERNEL_H__

float * canny_edge_detection(const float    *in,
                             const int      width,
		             const int      height,
			     const int      t1,
			     const int      t2,
			     const float    sigma);

# endif
