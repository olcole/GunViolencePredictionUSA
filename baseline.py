import math
from sklearn.metrics import mean_squared_error

def baselineModel(mean, test_data):
    means = list()

    for i in range((len(test_data))):
        means.append(mean)

    # i = 0
    # for x in test_data:
    #     print("PREDICTION: " + str(means[i]) + "     REAL: " + str(x))
    #     i += 1

    rmse = math.sqrt(mean_squared_error(test_data, means))
    #print('RMSE for raw incident counts: %.3f' % rmse)
    return rmse
