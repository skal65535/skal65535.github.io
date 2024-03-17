# <center><a href="./MASSIMINO_Pascal_Resume.pdf">Skal</a>'s corner: code snippets & demos</center>

My day-job is -amongst other things- to write open-sourced code (like [webp](https://github.com/webmproject/libwebp) or
[sjpeg](https://github.com/webmproject/sjpeg)). But! I also like to write
open-sourced code for fun :) Here are some side-projects demo pages:

## [Gaussian splats](https://skal65535.github.io/splats/index.html) (aka 3DGS)

   ![3DGS](https://skal65535.github.io/splats/splats.thumb.webp)

   Gaussian splats are the new cool in scene rendering, here's my WebGPU version!

## [Stippling](https://skal65535.github.io/stipple/index.html) toy

   ![stipple](https://skal65535.github.io/stipple/stipple.thumb.webp)

   Some LBG algorithm live action, with a lot of parameters to play with.

## [Ising Model](https://skal65535.github.io/ising/index.html) simulated with WebGPU

   ![Ising-Model](https://skal65535.github.io/ising/ising.thumb.webp)

   Using WebGPU compute-shader, we can perform Monte-Carlo sampling (aka Metropolis method)
   for the 3D cubic Ising model of magnetic spins, at interactive frame-rate!
   Back in the days, these sort of computations would require <i>days</i> of work on a Pentium!

## [Welzl algorithm](https://skal65535.github.io/convex_hull/index.html) demo

   A short demo in HTML showing convex hull and Welzl's
   smallest enclosing circle algo (in 2D).
   Its main interest is the code, not really the page in itself.

## [Kruskal algorithm](https://skal65535.github.io/network/kruskal.html) demo

   Very simple and elegant way to extract the Minimum-Spanning Tree out
   of a set of points!

## [Triangle-based compression](https://skal65535.github.io/triangle/index.html) demo

   Using triangulation + colormap to compress images into a very tight preview
   (compressed data is a base64 string).<br/>
   The decoder is ~400 lines of javascript + WebGL.

   See the [paper](http://arxiv.org/abs/1809.02257) presented at ICIP 2018.

## What's up with the root of [Trees](https://skal65535.github.io/tree/index.html)?

   ![tree](https://skal65535.github.io/tree/tree.thumb.webp)

   The root of a tree is not *that* particular, as seen with this
   [simple visualization](https://skal65535.github.io/tree/index.html)
   of a random tree that lets you pick any node as root.

## [Thinning](https://skal65535.github.io/thinning/index.html) algorithms

   ![thinning](https://skal65535.github.io/thinning/thinning.thumb.webp)

   Implementation of two thinning algorithms that extract skeletons from binarized images.

## [Exploring Difference Of Gaussian](https://skal65535.github.io/dog/dog.html)

   ![DoG](https://skal65535.github.io/dog/dog.thumb.webp)

   Difference of Gaussian operator
   (as described in the original [paper](https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf))
   has a lot of different parameters to play with.<br/>
   This HTML+WebGL page lets you do just that.

## [QR Code generator](https://skal65535.github.io/QR)

   ![QR Code](https://skal65535.github.io/QR/QRCode.thumb.webp)

   Embed pictures in QR codes

## [Particles](https://skal65535.github.io/particle_life/particle_life.html#91651088029) from [life_code](https://github.com/skal65535/life_code) project

   Explore randomly interacting particles. Click on 'Random exploration' button for interesting things to happen!

## My custom [MPU9255 Gyro / AK8963 Magnetometer / MCP2221 micro](https://github.com/skal65535/sklmpu9255) library

   I couldn't find a library for this IMU, so i rewrote one.
   I also rewrote some I2C functions for a MCP2221 USB<->I2C micro-controller, so i could play
   with my IMU directly from my MacBook Laptop. No more have to use a Raspberry Pi!

## Some shaders made with ShaderToy...

  * [Voronoi experiments](https://www.shadertoy.com/view/ftByDD)
  * [Fluid flow simulation](https://www.shadertoy.com/view/ft2czK)
  * [Group Velocity visualization](https://www.shadertoy.com/view/stcBDB)
  * [Mach cone visualization](https://www.shadertoy.com/view/slKfWR)
  * [Kelvin wake visualization](https://www.shadertoy.com/view/stGBWh)
  * [Rainbow visualization](https://www.shadertoy.com/view/NlyfRV)
  * [Deluge Simulator](https://www.shadertoy.com/view/slKfWc)

... and a [Minishader](https://skal65535.github.io/minishader/index.html) to
convert simple ShaderToy code to a standalone HTML + WebGL page.

![shaders](https://skal65535.github.io/common/deluge.thumb.webp)
