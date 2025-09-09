y_true = c(0, 0, 0, 0, 0, 0, 0, 1, 1, 1)
y_pred = c(0, 0, 0, 1, 1, 0, 1, 0, 0, 1)


noise_rate = 0.1


TP <- sum(y_true == 1 & y_pred == 1)
TN <- sum(y_true == 0 & y_pred == 0)
FP <- sum(y_true == 0 & y_pred == 1)
FN <- sum(y_true == 1 & y_pred == 0)


accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- TP / (TP + FP)
precision_over_noise <- ifelse(noise_rate != 0, precision - noise_rate, "unable to compute noise over precision")
recall <- TP / (TP + FN)
f1 <- 2 * precision * recall / (precision + recall)
beta <- 0.5
f05 <- (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)

expected_acc = 0.5
expected_precision = 0.25
expected_recall = 1/3
expected_f1 = 2/7
expected_f05 = 0.26
expected_precision_over_noise = 0.15

stopifnot(expected_acc == accuracy)
stopifnot(expected_precision == precision)

stopifnot(expected_recall == recall)
stopifnot(isTRUE(all.equal(expected_f1, f1)))
stopifnot(isTRUE(all.equal(expected_f05, f05, tolerance = 0.1)))
stopifnot(expected_precision_over_noise == precision_over_noise)
