#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <complex>
#include <utility>
#include <tuple>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor-io/ximage.hpp"

#include <boost/program_options.hpp>


void mandelbrot(
    std::pair<long, long> resolution, 
    std::pair<std::complex<double>, 
    std::complex<double>> bounds, 
    long max_iter,
    double r = 2.,
    int k = 2) {
    
    const double a0 = bounds.first.real();
    const double b0 = bounds.first.imag();
    const double a1 = bounds.second.real();
    const double b1 = bounds.second.imag();
    const long height = resolution.first;
    const long width = resolution.second;
    const long channels = 3;

    long n;
    double imag, real;
    auto I = std::complex<double>(0.0, 1.);

    std::complex<double> z0(0., 0.);
    std::complex<double> z(z0);
    std::complex<double> c;

    xt::xtensor<double, 1> xrange = xt::linspace<double>(a0, a1, width);
    xt::xtensor<double, 1> yrange = xt::linspace<double>(b0, b1, height);

    xt::xtensor<long, 2> counts = xt::zeros<long>({ height, width });
    xt::xtensor<long, 2> norms = xt::zeros<double>({ height, width });
    xt::xtensor<long, 2> args = xt::zeros<double>({ height, width });

    for (size_t i = 0; i < height; i++) {
        imag = yrange(i);

        std::cout << std::fixed << std::setprecision(2) << "\rRunning... " << 100.0 * (i + 1.0) / height << "%" << std::flush;

        for (size_t j = 0; j < width; j++) {
            real = xrange(j);

            n = 0;
            z = z0;
            c = std::complex<double>(real, imag);

            const bool period_one = std::abs(1. - std::sqrt(1. - 4. * c)) <= 1.;
            const bool period_two = std::abs(c + 1.) < 0.25;

            if (period_one || period_two) {
                n = max_iter;
            }
            else {
                while (n < max_iter && std::abs(z) < r) {
                    n++;
                    z = std::pow(z, k) + c;
                }
            }

            norms(i, j) = std::norm(z);
            args(i, j) = std::arg(z);
            counts(i, j) = n - std::log( std::log(std::norm(z))/std::log(r))/std::log(k);  // renormalized
        }
    }

    const double min_counts = xt::amin(counts)();
    const double max_counts = xt::amax(counts)();

    const double min_norms = xt::amin(norms)();
    const double max_norms = xt::amax(norms)();

    const double min_args = xt::amin(args)();
    const double max_args = xt::amax(args)();

    auto scaled_counts = (counts - min_counts) / (max_counts - min_counts);
    auto scaled_norms = (norms - min_norms) / (max_norms - min_norms);
    auto scaled_args = (args   - min_args  ) / (max_args   - min_args  );

    xt::dump_image("../../mandelbrot_counts.png", xt::cast<uint8_t>(255 * scaled_counts));
    xt::dump_image("../../mandelbrot_norms.png" , xt::cast<uint8_t>(255 * scaled_norms ));
    xt::dump_image("../../mandelbrot_args.png"  , xt::cast<uint8_t>(255 * scaled_args  ));

    xt::xtensor<uint8_t, 3> image = xt::zeros<uint8_t>({ height, width, channels });
    std::tuple<uint8_t, uint8_t, uint8_t> color;

    for (size_t i = 0; i < height; i++) {
        std::cout << std::fixed << std::setprecision(2) << "\rColoring... " << 100.0 * (i + 1.0) / height << "%" << std::flush;

        for (size_t j = 0; j < width; j++) {
            image(i, j, 0) = (uint8_t) 255 * scaled_counts(i, j);
            image(i, j, 1) = (uint8_t) 255 * scaled_args(i, j);
            image(i, j, 2) = (uint8_t) 255 * scaled_norms(i, j);
        }
    }

    xt::dump_image("../../mandelbrot_color.png", image);
}

namespace opt = boost::program_options;

int main(int argc, char** argv) {
    int height, width, max_iter;
    double xi, xf, yi, yf, r, p;


    opt::options_description params("Mandelbrot Set Parameters");

    params.add_options()
        ("help", "Show Usage")
        ("height,h", opt::value< int >(&height)->default_value(2160), "Image Height")  // 4320
        ("width,w", opt::value< int >(&width)->default_value(3840), "Image Width")  // 7680
        ("max_iter,m", opt::value< int >(&max_iter)->default_value(1000), "Maximum Number Of Iterations")
        ("xi", opt::value<double>(&xi)->default_value(-2.0), "Lower Real Bound")
        ("xf", opt::value<double>(&xf)->default_value(1.0), "Upper Real Bound")
        ("yi", opt::value<double>(&yi)->default_value(-1.5), "Lower Imaginary Bound")
        ("yf", opt::value<double>(&yf)->default_value(1.5), "Upper Imaginary Bound")
        ("radius,r", opt::value<double>(&r)->default_value(2.), "Bail-Out Radius")
        ("power,p",  opt::value<double>(&p)->default_value(2.), "Polynomial Power Of The Logistic Map (Classic Mandelbrot Set: p = 2)")
        ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, params), vm);

    if (vm.count("help")) {
        std::cout << params << std::endl;
        return 1;
    }
    else {
        opt::notify(vm);
        mandelbrot(std::make_pair(height, width), std::make_pair(std::complex<double>(xi, yi), std::complex<double>(xf, yf)), max_iter, r, p);
        return 0;
    }
}
