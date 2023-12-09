function plot_data(x,y)
 subplot(1,3,1);
 plot(x(:,1),y,'*r','linewidth',0.2);
 xlabel('PM2.5');
 ylabel('AQI');
 subplot(1,3,2);
 plot(x(:,2),y,'+b','linewidth',0.2);
 xlabel('PM10');
 ylabel('AQI');
 subplot(1,3,3);
 plot(x(:,3),y,'xc','linewidth',0.2);
 xlabel('NO');
 ylabel('AQI');

 figure(2)
 subplot(1,3,1);
 plot(x(:,4),y,'*r','linewidth',0.2);
 xlabel('NO2');
 ylabel('AQI');
 subplot(1,3,2);
 plot(x(:,5),y,'+b','linewidth',0.2);
 xlabel('NOx');
 ylabel('AQI');
 subplot(1,3,3);
 plot(x(:,6),y,'xc','linewidth',0.2);
 xlabel('NH3');
 ylabel('AQI');

 figure(3);
 subplot(1,3,1);
 plot(x(:,7),y,'*r','linewidth',0.2);
 xlabel('CO');
 ylabel('AQI');
 subplot(1,3,2);
 plot(x(:,8),y,'+b','linewidth',0.2);
 xlabel('SO2');
 ylabel('AQI');
 subplot(1,3,3);
 plot(x(:,9),y,'xc','linewidth',0.2);
 xlabel('O3');
 ylabel('AQI');

 figure(4)
 subplot(1,3,1);
 plot(x(:,10),y,'*r','linewidth',0.2);
 xlabel('BENZENE');
 ylabel('AQI');
 subplot(1,3,2);
 plot(x(:,11),y,'+b','linewidth',0.2);
 xlabel('TOLUENE');
 ylabel('AQI');
 subplot(1,3,3);
 plot(x(:,12),y,'xc','linewidth',0.2);
 xlabel('XYLENE');
 ylabel('AQI');
end
