from utils import *

# Load submissions
# xgb_1400 = get_submission('submissions/xgb_1400.csv')
# xgb_1450 = get_submission('submissions/xgb_1450.csv')
# xgb_1500 = get_submission('submissions/xgb_1500.csv')
xgb = get_submission('submissions/xgb_average.csv')
lasagne = get_submission('submissions/lasagne2.csv')
h2o = get_submission('submissions/h2o_scaled.csv')

# Append scores
# xgb_average = (xgb_1400 + xgb_1450 + xgb_1500) / 3
predictions = 1.0 / 0.462 * xgb + 1.0 / 0.449 * h2o + 1.0 / 0.464 * lasagne

# Save submission
save_submission(predictions, 'submissions/xgb_lasagne_h2o_weighted.csv')
