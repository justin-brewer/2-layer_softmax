function mnist = gradientProp()

    % mnist

    load('data.mat');
    % load('dat.mat');
    
    % num training samples
    n = 2000;
    S = mdata.train.im;
%     S = mdata.train.im(:, 1:n);
 % image size + 1
    [imsize, ni] = size(S);
    S = [S; ones(1, ni)];
    [imsize, ni] = size(S);
    
    O = mdata.train.lab;
%     O = mdata.train.lab(:, 1:n);
    
    % output size + 1
    osize = size(O, 1);

    % num test samples
    tn = 10000;
    St = mdata.test.im(:, 1:tn);
    St = [St; ones(1, tn)];
    Ot = mdata.test.lab(:, 1:tn);
      
    
   
   
    
    
    
    nrhidden = 1000;


    W = (rand(nrhidden,imsize)-.5)*.2;
%     W(:,end) = W(:,end)/10;
    % wh = struct('w0', W);

    %V = ones(10,785).*wts;
    V = (rand(osize, nrhidden + 1)-.5)*.2;
    % vh = struct('v0', V);

   
    gam1 = 0.001;
    gam2 = 0.001;

    for i = 1:15000
        
%         s = S(:, i:(n + i - 1));
%         o = O(:, i:(n + i - 1));
        
        [s, idx] = datasample(S, n, 2);
        o = O(:, idx);
        
%         r = W*S;
        r = W*s;
        h = log(1 + exp(r));
        hT = h';
        % hi = [h; ones(n)];
        f = V*[h; ones(1, n)];
        %f = V*h;
        
        f_max = max(f);
        fm = f - ones(10, 1)*f_max;
        q = exp(fm); % 10,n
        q_sum = sum(q); % 1,n
        q = q./(ones(osize, 1)*q_sum);

%         ln = -(1/n)*sum(sum(O.*log(q)));
        ln = -(1/n)*sum(sum(o.*log(q)));
    
        % n,o
%         dl_df = -(O - q)';
        dl_df = -(o - q)';
        
        % h+1,0
        df_dV = [hT, ones(n,1)];
        
        % 10,1000 ; 1000,1000
        dl_dV = dl_df'*df_dV;

        df_dh = V(:, 1:(end-1));

        dh_dr = 1./(1 + exp(-r));


        %       200,10;10,785; 785,200
        dl_dr = (dl_df*df_dh)'.*dh_dr;


        % 785,785;     200,785
%         dl_dW = dl_dr*S';
        dl_dW = dl_dr*s';

        W = W - gam1.*dl_dW;
        V = V - gam2.*dl_dV;
        
        if ln < 0.01
            break;
        end

%         whs = ['w', num2str(i)];
%         vhs = ['v', num2str(i)];

%         wh.(whs) = W;
%         vh.(vhs) = V;

        if mod(i, 100) == 0
            msg = ['error after ', num2str(i), ': ', num2str(ln)];
            disp(msg);
            
%             wnorm = sum(sum(dl_dW.*dl_dW));
%             vnorm = sum(sum(dl_dV.*dl_dV));
%             msg2 = ['; W norm: ', num2str(wnorm), ' ; V norm: ', num2str(vnorm)];
%             msgd = [msg, msg2];
%             disp(msgd);
        end
        
        if mod(i, 100) == 0
            gam1 = gam1*.91;
            gam2 = gam2*.91;
        end
    end % END MAIN LOOP
    
    r = W*S;
    h = log(1 + exp(r));
    hT = h';
    % hi = [h; ones(n)];
    f = V*[h; ones(1, ni)];
    %f = V*h;

    f_max = max(f);
    fm = f - ones(10, 1)*f_max;
    q = exp(fm); % 10,n
    q_sum = sum(q); % 1,n
    q = q./(ones(osize, 1)*q_sum);

%         ln = -(1/n)*sum(sum(O.*log(q)));
%     ln = -(1/ni)*sum(sum(O.*log(q)));
%     msg = ['final ln val: ', num2str(ln)];
%     disp(msg);
    
    yd = zeros(osize, ni);
    for i = 1:ni
        [val, idx] = max(q(:, i));
        yd(idx, i) = 1;
    end
    
    err = (O - yd).^2;
    err = sum(max(err))/ni;
    msg = ['training err: ', num2str(err)];
    disp(msg);
    
    
    
    
    r = W*St;
    h = log(1 + exp(r));
    hT = h';
%     f = V*h;
    f = V*[h; ones(1, tn)];
    f_max = max(f);
    fm = f - ones(osize, 1)*f_max;
%     fm = f - ones((nrhidden + 1), 1)*f_max;
    q = exp(fm); % 10,n
    q_sum = sum(q); % 1,n
    q = q./(ones(osize, 1)*q_sum);
%     ln_t = -(1/n_?)*sum(sum(Ot.*log(q)));
    
    ydt = zeros(osize, tn);
    for i = 1:tn
        [val, idx] = max(q(:, i));
        ydt(idx, i) = 1;
    end
    
    errt = (Ot - ydt).^2;
    
    errt = sum(max(errt))/tn;
    
    msg = ['test err: ', num2str(errt)];
    disp(msg);
    
    

%     a = 2;
end

