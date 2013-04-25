# The Linear Regression Algorithm
# on input of two dimensions
#
# Katie Cunningham
# Developed with Python 2.7


# imports
from numpy import zeros, ones, array, sign, dot, mean, linspace, vstack
from numpy.random import random, randint, permutation, uniform
from numpy.linalg import norm, inv, lstsq
from scipy import stats
from pylab import figure, plot, xlabel, ylabel, xlim, ylim, title
from pylab import draw, show, pause

# globals
VISUALIZE = True


def plot_pts(pts, pts_labels=None, plus1_format='co', minus1_format='yo',
             other_format='ko'):
    """ Plots 2nd and 3rd cols of pts as x and y. If pts_labels is not None,
    coloring is based on pts_labels, and by default +1s are plus1_format, -1s
    are minus1_format, and other values are other_format. If pts_labels is
    None, all points are other_format.

    """

    if pts_labels is None:
        plot(pts[:,1], pts[:,2], other_format)

    else:
        for i in xrange(len(pts_labels)):
            if pts_labels[i] == 1:
                plot(pts[i,1], pts[i,2], plus1_format)
            elif pts_labels[i] == -1:
                plot(pts[i,1], pts[i,2], minus1_format)
            else:
                plot(pts[i,1], pts[i,2], other_format)


def plot_from_vector(v, line_format='g-'):

    if v[2] != 0:
        plot([-1, 1],
             [(v[1]+ -v[0])/v[2], (-v[1] + -v[0])/v[2]],
             line_format)


def generate_random_data(N, x_1_range, x_2_range):
    """ Generate a random data set sof size N uniformly distributed within
    x_1_range and x_2_range

    """

    x_1s = uniform(x_1_range[0], x_1_range[1], N)
    x_2s = uniform(x_2_range[0], x_2_range[1], N)

    return x_1s, x_2s


def classify_pts(f_w, pts):
    """ Return an array of 1s and -1s based on which side of f_w the
    corresponding member of pts is on.

    """

    N = len(pts[:,0])

    labels = zeros(N)

    for i in xrange(N):
        label = sign(dot(f_w, pts[i,:]))
        if label == 0:
            label = 1

        labels[i] = label

    return labels


def flip_some(array, fraction):
    """ Flip fraction of array's signs.
    Useful for adding random error.

    """

    num_to_flip = fraction * len(array)
    ii_to_flip = permutation(range(len(array)))[:num_to_flip]
    for ii in ii_to_flip:
        array[ii] = array[ii] * -1


def new_space(x_1s, x_2s, transform=lambda x_1, x_2 : [1, x_1, x_2]):
    """ Tranform data points into a space defined by transform.
    transform should be a function that takes two arguments, x_1 and x_2,
    and returns an array-like object that represents the new space.

    """

    transformed = zeros((len(x_1s),len(transform(1,1))))

    ii = 0
    for x_1, x_2 in zip(x_1s, x_2s):
        #print ii, x_1, x_2
        transformed[ii,:] = array(transform(x_1, x_2))
        ii += 1

    return transformed


def linregress(x, y):
    """ Find the linear regression of the given points.
    x should be a N by L array, where N is number of points and L is dimension
    of the points. y should be a one dimentional array of length N."""

    # Find the pseudo-inverse
    x_dagger = dot(inv(dot(x.T, x)), x.T)

    # Use that to find g_w, the vector representing the learned function
    g_w = dot(x_dagger, y)
    return g_w


def find_prob_fx_neq_gx(f_w, g_w):
    """ Compare f and g to find Pr(f(x) != g(x)).
    This is the proportion of the space where the f and g do not overlap.

    """

    density = 100
    xx = linspace(-1, 1, density)
    yy = linspace(-1, 1, density)
    wrong = 0
    for ii in xx:
        for jj in yy:
            test_f = dot(f_w, [1,ii,jj])
            test_g = dot(g_w, [1,ii,jj])

            if sign(test_f) != sign(test_g):
                wrong = wrong + 1
                if VISUALIZE:
                    plot(ii,jj,'c|')

    return float(wrong) / (density * density)


def find_E_pts(f_w, g_w, pts, pts_labels):
    """Find the percentage of pts that are misclassified by the learned fxn.
    f_w is the target function vector, g_w is the learned function."""

    N = len(pts[:,0])

    wrong = 0
    for i in xrange(N):
        test_g = dot(g_w, pts[i,:])

        if sign(test_g) != pts_labels[i]:
            wrong = wrong + 1
            if VISUALIZE:
                plot(pts[i,1],pts[i,2],'m*')

    draw()

    return float(wrong) / N


def main():
    """ Estimate a circular function by linear regression on transformed data
    with a 10% error rate.

    """

    # training set size
    N = 1000

    # testing set size
    N_test = 1000

    # data point range
    x_1_range = [-1,1]
    x_2_range = [-1,1]

    # The target function
    f = lambda x_1,x_2 : sign(x_1**2 + x_2**2 - 0.6)
    # Vector version of the target function
    f_w = array([-0.6, 0, 0, 0, 1, 1])

    # The space transformation
    transform = lambda x_1,x_2 : array([1,x_1, x_2, x_1*x_2, x_1*x_1, x_2*x_2])

    num_experiment_repeats = 3

    # Keep track of stats over experiments
    #prob_wrong_results = zeros(num_experiment_repeats)
    E_in_results = zeros(num_experiment_repeats)
    E_out_results = zeros(num_experiment_repeats)
    g_w_results = zeros((num_experiment_repeats, len(transform(1,1))))

    for repeat_i in xrange(num_experiment_repeats):

        # Generate some data points
        x_1s, x_2s = generate_random_data(N, x_1_range, x_2_range)

        # The y values will have some random error
        #   First, find their real classification based on the target function
        x_trans = new_space(x_1s, x_2s, transform=transform)
        real_labels = classify_pts(f_w, x_trans)

        plot_pts(x_trans, real_labels)

        show()

        y = real_labels.copy()
        #   Second, give them random error
        flip_some(y, 0.1)

        # Plot points to learn from
        if VISUALIZE:
            figure()
            xlim(x_1_range)
            ylim(x_2_range)
            xlabel("x_1")
            ylabel("x_2")
            title("data to learn from")
            plot_pts(x_trans, y)
            pause(0.5)

        # Use linear regression on a transformed space to learn the function
        g_w = linregress(x_trans, y)
        # Save the learned function
        g_w_results[repeat_i,:] = g_w
        print g_w

        # Calculate different measures of error

        #prob_wrong_results[repeat_i] = find_prob_fx_neq_gx(f_w, g_w)

        E_in_results[repeat_i] = find_E_pts(f_w, g_w, x_trans, y)
        title("misclassified points are magenta")


        #out_pts_x_1s, out_pts_x_2s = generate_random_data(N_test, x_1_range,
        #                                                        x_2_range)
        #out_pts_trans = new_space(out_pts_x_1s, out_pts_x_2s, transform)
        #real_out = classify_pts(f_w, new_space(out_pts_x_1s, out_pts_x_2s,
        #                                       transform=transform))
        #y_out = real_out.copy()
        #flip_some(y_out, 0.1)
        #if VISUALIZE:
        #    pause(0.5)
        #    plot_pts(out_pts_trans, y_out)
        #    pause(0.5)
        #E_out_results[repeat_i] = find_E_pts(f_w, g_w, out_pts_trans, y_out)

        show()

    #av_prob_wrong = mean(prob_wrong_results)
    #print "av_prob_wrong=",av_prob_wrong
    av_E_in = mean(E_in_results)
    print "av_E_in=",av_E_in
    av_E_out = mean(E_out_results)
    print "av_E_out=",av_E_out
    av_g_w = mean(g_w_results, axis=0)
    print "av_g_w=",av_g_w





if __name__ == '__main__':
    main()

