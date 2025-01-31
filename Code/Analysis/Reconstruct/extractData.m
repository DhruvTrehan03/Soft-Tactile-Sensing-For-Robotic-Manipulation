function [logtimes, presstimes, responses] = extractData(boardfile, pressfile)
    logtimes(2) = datetime();
    presstimes(2) = datetime();
    
    fid = fopen("C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Code\Analysis\Reconstruct\Data\"+boardfile+".log");
    n = 0;
    tline = fgetl(fid);
    while ischar(tline)
      tline = fgetl(fid);
      n = n+1;
    end
    fclose(fid);
    fid = fopen("Data/"+boardfile+".log");

    for i = 1:5
        fgetl(fid);
    end

    for i = 1:n-6
        line = fgetl(fid);
        logtimes(i) = datetime(line(2:24),'InputFormat','yyyy-MM-dd HH:mm:ss.SSS');
        stringvalues = split(line(27:end-2),',');
        for j = 1:length(stringvalues)
            responses(i, j) = str2double(stringvalues{j});
        end
    end
    fclose(fid);
    
    fid = fopen("Data/"+pressfile+".txt");
    n = 0;
    tline = fgetl(fid);
    while ischar(tline)
      tline = fgetl(fid);
      n = n+1;
    end
    fclose(fid);
    fid = fopen("Data/"+pressfile+".txt");
    for i = 1:n
        line = fgetl(fid);
        presstimes(i) = datetime(line,'InputFormat','HH:mm:ss');
        presstimes(i).Day = logtimes(i).Day;
        presstimes(i).Month = logtimes(i).Month;
        presstimes(i).Year = logtimes(i).Year;
    end
    fclose(fid);
end