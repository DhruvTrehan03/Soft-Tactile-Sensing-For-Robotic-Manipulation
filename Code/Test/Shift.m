% Main Script

data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
data_hom = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
data_hom = circshift(data_hom,32,1);
data_objs = circshift(data_objs,32,1);
data_hom = rem_0(data_hom);
data_objs = rem_0(data_objs);


mdl = load("Simulation\Model.mat","mdl").mdl;

subplot(3,1,1)
plot(data_hom)
subplot(3,1,2)
plot(data_objs)
subplot(3,1,3)
plot(abs(data_objs-data_hom))

function[Data] = rem_0(Data)

% Remove columns with all NaN values
columns_with_zeros = all(Data==0);  % Identify columns that are entirely NaN
Data = Data(:, ~columns_with_zeros);       % Remove those columns
end