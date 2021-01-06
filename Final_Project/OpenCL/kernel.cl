#define MAX_BRIGHTNESS 255

__kernel void calculate_G(__global float * G,
			  __global float * G_x,
			  __global float * G_y,
			  int width,
			  int height)
{
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int idx = thisY * width + thisX;

    if (thisX != 0 && thisX != width-1 && thisY != 0 && thisY != height-1){
	G[idx] = (float)hypot(G_x[idx], G_y[idx]);
    }
    else{
	G[idx] = 1;
    }
}

__kernel void non_maximum_sup(__global float * nms,
			      __global float * G,
			      __global float * G_x,
			      __global float * G_y,
			      int width,
			      int height)
{
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int idx = thisY * width + thisX;

    if (thisX != 0 && thisX != width-1 && thisY != 0 && thisY != height-1){
       	const int nn = idx - width;
       	const int ss = idx + width;
       	const int ww = idx + 1;
       	const int ee = idx - 1;
       	const int nw = nn + 1;
       	const int ne = nn - 1;
       	const int sw = ss + 1;
       	const int se = ss - 1;
       	const float dir = (float) (fmod(atan2(G_y[idx], G_x[idx]) + M_PI, M_PI) / M_PI) * 8;

        if (((dir <= 1 || dir > 7) && G[idx] > G[ee] && G[idx] > G[ww]) || // 0 deg
            ((dir > 1 && dir <= 3) && G[idx] > G[nw] && G[idx] > G[se]) || // 45 deg
            ((dir > 3 && dir <= 5) && G[idx] > G[nn] && G[idx] > G[ss]) || // 90 deg
            ((dir > 5 && dir <= 7) && G[idx] > G[ne] && G[idx] > G[sw]))   // 135 deg
	    nms[idx] = G[idx];
	else
	    nms[idx] = 0;
    }
    else{
	nms[idx] = 1;
    }
}

__kernel void convolution(__global float * in,
			  __global float * out,
			  __global float * kern,
			  int nx,
			  int ny,
			  int kn,
			  int normalize)
{
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int idx = thisY * nx + thisX;

    const int khalf = kn / 2;
    float min = 0.5;
    float max = 254.5;
    float pixel = 0.0;
    size_t c = 0;
    int i, j;

    if (thisX >= khalf && thisX < nx-khalf && thisY >= khalf && thisY < ny-khalf){
	pixel = c = 0;

        for (j = -khalf; j <= khalf; j++)
          for (i = -khalf; i <= khalf; i++)
            pixel += in[(thisY - j) * nx + thisX - i] * kern[c++];

        if (normalize == 1)
          pixel = MAX_BRIGHTNESS * (pixel - min) / (max - min);

        out[idx] = (float) pixel;
    }
    else{
	out[idx] = 0;
    }
}

__kernel void threshold(__global float * nms,
			__global float * thre,
			int width,
			int t2)
{
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int idx = thisY * width + thisX;

    if (nms[idx] >= t2){
	thre[idx] = MAX_BRIGHTNESS;
    }
    else{
	thre[idx] = 0;
    }
}

__kernel void hysteresis(__global float * nms,
			 __global float * thre,
			 __global float * hyster,
			 int width,
			 int height,
			 int t1,
			 int t2)
{
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int idx = thisY * width + thisX;

    const int nn = idx - width;
    const int ss = idx + width;
    const int ww = idx + 1;
    const int ee = idx - 1;
    const int nw = nn + 1;
    const int ne = nn - 1;
    const int sw = ss + 1;
    const int se = ss - 1;

    hyster[idx] = thre[idx];

    if (thisX != 0 && thisX != width-1 && thisY != 0 && thisY != height-1){
	if (t1 < nms[idx] && nms[idx] < t2){
	    if (thre[ee] != 0 || thre[ww] != 0 ||
		thre[nn] != 0 || thre[ss] != 0 ||
		thre[ne] != 0 || thre[nw] != 0 ||
		thre[se] != 0 || thre[sw] != 0){
		hyster[idx] = MAX_BRIGHTNESS;
	    }
	}
    }
}
