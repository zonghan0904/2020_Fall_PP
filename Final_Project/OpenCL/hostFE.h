# ifndef __KERNEL_H__
# define __KERNEL_H__
# include <CL/cl.h>

float * canny_edge_detection(const float    *in,
                             const int      width,
		             const int      height,
			     const int      t1,
			     const int      t2,
			     const float    sigma,
			     cl_device_id   *device,
			     cl_context	    *context,
			     cl_program	    *program);

# endif
