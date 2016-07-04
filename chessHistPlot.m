load('test_trans_error5.txt');

figure;
subplot(2,1,1)
hist(test_trans_error5,100);
title('within 5 degrees')
xlabel('meters')
ylabel('degrees')

subplot(2,1,2)
hist(test_trans_error10,100);
xlabel('meters')
ylabel('degrees')
title('within 10 degrees')
