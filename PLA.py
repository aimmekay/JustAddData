# The Perceptron Learning Algorithm
#
# Developed with Python 2.7

def PLA(g_w, training_pts, training_pts_labels):
    """Run PLA starting from g_w"""
    # Start hypothesis function out as all zeros
    g_w_prev = g_w

    N = len(training_pts_labels)

    num_iter = 0
    g_pts_labels = zeros(N)

    while True:

        # Check if all the points are classified and break if so
        for i in range(N):
            if dot(g_w, training_pts[i,:]) >= 0:
                g_pts_labels[i] = 1
            else:
                g_pts_labels[i] = -1

        if sum(g_pts_labels == training_pts_labels) == N:
            break

        # plot current g_w
        #if g_w[2] != 0:#plot([-1, 1], [(g_w[1]-g_w[0])/g_w[2], (-g_w[1]-g_w[0])/g_w[2]],'k-')

        # color points as classified by g
        #for i in range(N):
        #    if dot(g_w, training_pts[i,:]) >= 0:
               #plot(training_pts[i,1], training_pts[i,2], 'sc')
        #    else:
               #plot(training_pts[i,1], training_pts[i,2], 'sy')

        # Now learn something

        # pick a misclassified point
        index_miscl = randint(N)
        while sign(dot(f_w, training_pts[index_miscl,:])) == sign(dot(g_w, training_pts[index_miscl,:])):
            index_miscl = randint(N)

        # color it magenta
        x_miscl = training_pts[index_miscl,:]
       #plot(x_miscl[1], x_miscl[2], 'sm')

        #uncolor the old g (unless g_w(3) == 0)
        #if g_w[2] != 0:#plot([-1, 1], [(g_w[1]-g_w[0])/g_w[2], (-g_w[1]-g_w[0])/g_w[2]],'w-')

        y_miscl = sign(dot(f_w, x_miscl))
        if y_miscl == 0:
            y_miscl = 1

        # and update the weight vector
        g_w_prev = g_w
        g_w = g_w + dot(y_miscl, x_miscl.T)

        # uncolor the tested point
       #plot(x_miscl[1], x_miscl[2], 'sb')
       #pause(0.5)
        num_iter = num_iter + 1

    return (g_w, num_iter)
