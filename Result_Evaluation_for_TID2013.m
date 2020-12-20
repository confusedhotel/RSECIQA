%%
sc = tid20131(:,1);
mos = tid20131(:,2);
figure(1)
scatter(sc, mos, 'k');
axis([0 9 0 9]);
srcc=num2str(corr(sc, mos, 'type', 'Spearman'));
plcc=num2str(corr(sc, mos, 'type', 'Pearson'));
title(['误差对比 srcc= ' srcc]);
xlabel('客观评分');
ylabel('MOS');
legend('失真图像');
hold on 