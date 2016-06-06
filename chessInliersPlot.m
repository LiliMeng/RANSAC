clear all
clc
fileID1= fopen('/home/lili/PatternRecognition/RANSAC/chess_inliers_result/inliers_tran_rot_error_200images.txt');
C = textscan(fileID1, '%s %s %s %s');


[N,d]=size(C{1});

m1=str2mat(C{1});
m2=str2mat(C{2});
m3=str2mat(C{3});
m4=str2mat(C{4});


for i = 1:N
    frame_num(i,:)=str2double(m1(i,:));
    inliers(i,:)=str2double(m2(i,:));
    trans_error(i,:)=str2double(m3(i,:));
    rot_error(i,:)=str2double(m4(i,:));
end

figure 

plot(inliers, trans_error, 'o');
t=title('inliers number Vs translational error ');
%le=legend('pchip data(16Hz)','pchip curve','raw data curve','raw data(18~20Hz)');
%set(le, 'FontSize', 16);
set(t, 'FontSize', 20);
xl=xlabel('inliers number');
yl=ylabel('translational error(m)');
set(xl, 'FontSize', 18);
set(yl, 'FontSize', 18);
axis([0 1000 0 1]);
figure 

plot(inliers, rot_error, '*');
t=title('inliers number Vs rotational error ');
%le=legend('pchip data(16Hz)','pchip curve','raw data curve','raw data(18~20Hz)');
%set(le, 'FontSize', 16);
set(t, 'FontSize', 20);
xl=xlabel('inliers number');
yl=ylabel('rotational error(degree)');
set(xl, 'FontSize', 18);
set(yl, 'FontSize', 18);
axis([0,1000,0,15])

figure
plot(trans_error, rot_error, 'b*');
t=title('translational Vs rotational error ');
%le=legend('pchip data(16Hz)','pchip curve','raw data curve','raw data(18~20Hz)');
%set(le, 'FontSize', 16);
set(t, 'FontSize', 20);
xl=xlabel('translational error(m)');
yl=ylabel('rotational error(degree)');
set(xl, 'FontSize', 18);
set(yl, 'FontSize', 18);

figure
plot(frame_num, trans_error,'*');
t=title('frame num Vs translational error ');
%le=legend('pchip data(16Hz)','pchip curve','raw data curve','raw data(18~20Hz)');
%set(le, 'FontSize', 16);
set(t, 'FontSize', 20);
xl=xlabel('frame num');
yl=ylabel('translational error(m)');
set(xl, 'FontSize', 18);
set(yl, 'FontSize', 18);
ylim([0,1])
%axis([0,100,0,1])
