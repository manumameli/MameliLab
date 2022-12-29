% This code is used to extract features, process dimensonality reduction and generate classifier based on calcium activity and behavioural events.
% Created by Benoit Girard
% Input:
% - label_event: matrix with numerical id corresponding to each behavioural events.
% - dataset: matrix of calcium activity
% Process:
% 1 - features extraction
% 2 - data normalization
% 3 - event prediction 
% 4 - dimensionality reduction


close all
clear all

rng('default') 
sampling_rate = 20;
resolution = 1./sampling_rate;

%% Features extraction
activity_increase = double(dataset > 1.96);
tot_duration_event_all = [];
mean_duration_event_all = [];
nb_event_all = [];
median_duration_event_all = [];
CV_duration_event_all = [];
skew_duration_event_all = [];
kurtosis_duration_event_all = [];
for n = 1:size(dataset,1)
    activity_increase_temp = [0 activity_increase(n,:)];
    activity_increase_temp = diff(activity_increase_temp);
    idx_start = find(activity_increase_temp==1);
    idx_end = find(activity_increase_temp==-1);
    if length(idx_start) > length(idx_end)
        idx_end = [idx_end length(activity_increase_temp)];
    end
    duration = idx_end-idx_start;
    tot_duration_event = (1 ./ (size(dataset,2).*resolution)) .* (nansum(duration).*resolution);
    mean_duration_event = nanmean(duration).*resolution; 
    nb_event = length(duration); 
    if isnan(tot_duration_event)
        tot_duration_event = 0;
    end
    if isnan(mean_duration_event)
        mean_duration_event = 0;
    end    
    if isnan(nb_event)
        nb_event = 0;
    end
    median_duration_event = nanmedian(duration); 
    CV_duration_event = nanstd(duration)/nanmean(duration);
    skew_duration_event = skewness(duration);
    kurtosis_duration_event = kurtosis(duration);
    if isnan(median_duration_event)
        median_duration_event = 0;
    end
    if isnan(CV_duration_event)
        CV_duration_event = 0;
    end    
    if isnan(skew_duration_event)
        skew_duration_event = 0;
    end
    if isnan(kurtosis_duration_event)
        kurtosis_duration_event = 0;
    end
    tot_duration_event_all = [tot_duration_event_all;tot_duration_event];
    mean_duration_event_all = [mean_duration_event_all;mean_duration_event];
    nb_event_all = [nb_event_all;nb_event];
    median_duration_event_all = [median_duration_event_all;median_duration_event];
    CV_duration_event_all = [CV_duration_event_all;CV_duration_event];
    skew_duration_event_all = [skew_duration_event_all;skew_duration_event];
    kurtosis_duration_event_all = [kurtosis_duration_event_all;kurtosis_duration_event];
end
dataset_temp = dataset(:,30:49);
mean_amplitude_all = nanmean(dataset_temp,2);
median_amplitude_all = nanmedian(dataset_temp,2);
CV_amplitude_all = nanstd(dataset_temp,1,2)./nanmean(dataset_temp,2);
skew_amplitude_all = skewness(dataset_temp,1,2);
kurtosis_amplitude_all = kurtosis(dataset_temp,1,2);

data_features = [tot_duration_event_all mean_duration_event_all nb_event_all median_duration_event_all CV_duration_event_all skew_duration_event_all kurtosis_duration_event_all mean_amplitude_all median_amplitude_all CV_amplitude_all skew_amplitude_all kurtosis_amplitude_all];

%% Data normalization
for i = 1:size(data_features,2)
    data_features(:,i) = zscore(data_features(:,i));
    data_features(:,i) = rescale(data_features(:,i),-1,1);
end

%% UMAP
[reduction, ~]=run_umap(data_features,'n_neighbors',180,'min_dist',0.79,'randomize',false,'n_components',3); 
close all
figure(11);clf;hold on;
tot_label = unique(label_event);
tot_color = colormap(jet(length(tot_label)));
for i = 1:length(tot_label)-1
    hold on;
    scatter3(reduction(find(label_event==tot_label(i)),1),reduction(find(label_event==tot_label(i)),2),reduction(find(label_event==tot_label(i)),3),10,tot_color(i,:),'filled')
end

%% Classifier
dataset_matlab_app = [label_event data_features];

model_KNN_1 = trainedModel.ClassificationKNN;
[Pred,PredScore] = predict(model_KNN_1,data_features);


