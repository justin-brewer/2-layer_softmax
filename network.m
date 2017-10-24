function mnist = myNetwork()
    strt = datetime('now');
    load('data.mat');
    
    S = mdata.train.im;
    [imsize, ni] = size(S);
    S = [S; ones(1, ni)];
    [imsize, ni] = size(S);
    O = mdata.train.lab;
    osize = size(O, 1);
    tn = 10000;
    St = mdata.test.im(:, 1:tn);
    St = [St; ones(1, tn)];
    Ot = mdata.test.lab(:, 1:tn);
    nrhidden = 800;

%     sd = sqrt(2/imsize);
    sd = 0.01;
    W = sd.*randn(nrhidden, imsize);
    V = sd.*randn(osize, nrhidden + 1);

    gam1 = 0.001;
    gam2 = 0.001;
    
    beta = 0.0005;
    
    B1_W = 0.9;
    B1_V = 0.9;
    
    B2_W = 0.999;
    B2_V = 0.999;
    
    eps = 10e-8;
    
    m_W = 0;
    m_V = 0;
    
    v_W = 0;
    v_V = 0;

    
    nit = 4000;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:nit
        n = i;
        [s, idx] = datasample(S, n, 2);
        o = O(:, idx);
        r = W*s;
        h = log(1 + exp(r));
        hT = h';
        f = V*[h; ones(1, n)];
        f_max = max(f);
        fm = f - ones(10, 1)*f_max;
        q = exp(fm);
        q_sum = sum(q);
        q = q./(ones(osize, 1)*q_sum);
        
        L2W = .5*sum(sum(W.*W));
        L2V = .5*sum(sum(V.*V));
        L2 = L2W + L2V;
        
        ln = -(1/n)*sum(sum(o.*log(q))) + beta*L2;
        
        dl_df = -(o - q)';
        df_dV = [hT, ones(n,1)];
        dl_dV = dl_df'*df_dV + beta.*V;
        df_dh = V(:, 1:(end-1));
        dh_dr = 1./(1 + exp(-r));
        dl_dr = (dl_df*df_dh)'.*dh_dr;
        dl_dW = dl_dr*s' + beta.*W;
        
        m_W = B1_W * m_W + (1 - B1_W) * dl_dW;
        m_V = B1_V * m_V + (1 - B1_V) * dl_dV;
        
        v_W = B2_W * v_W + (1 - B2_W) * (dl_dW .* dl_dW);
        v_V = B2_V * v_V + (1 - B2_V) * (dl_dV .* dl_dV);
        
        mbc_W = m_W/(1 - B1_W^i);
        mbc_V = m_V/(1 - B1_V^i);
        
        vbc_W = v_W/(1 - B2_W^i);
        vbc_V = v_V/(1 - B2_V^i);
        
        W = W - gam1 .* (mbc_W./(sqrt(vbc_W) + eps));
        V = V - gam2 .* (mbc_V./(sqrt(vbc_V) + eps));

        if mod(i, 100) == 0
            msg = ['err(', num2str(i), '): ', num2str(ln)];
            disp(msg);
        end
    end % END MAIN LOOP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    r = W*S;
    h = log(1 + exp(r));
    hT = h';
    f = V*[h; ones(1, ni)];
    f_max = max(f);
    fm = f - ones(10, 1)*f_max;
    q = exp(fm); % 10,n
    q_sum = sum(q); % 1,n
    q = q./(ones(osize, 1)*q_sum);
    ln = -(1/ni)*sum(sum(O.*log(q)));
    msg = ['final ln val: ', num2str(ln)];
    disp(msg);
    yd = zeros(osize, ni);
    for i = 1:ni
        [val, idx] = max(q(:, i));
        yd(idx, i) = 1;
    end
    err = (O - yd).^2;
    err = sum(max(err))/ni;
    acc = (1 - err)*100;
    msg = ['training accuracy: ', num2str(acc)];
    disp(msg);
   
    r = W*St;
    h = log(1 + exp(r));
    hT = h';
    f = V*[h; ones(1, tn)];
    f_max = max(f);
    fm = f - ones(osize, 1)*f_max;
    q = exp(fm); % 10,n
    q_sum = sum(q); % 1,n
    q = q./(ones(osize, 1)*q_sum);
    ydt = zeros(osize, tn);
    for i = 1:tn
        [val, idx] = max(q(:, i));
        ydt(idx, i) = 1;
    end
    errt = (Ot - ydt).^2;
    errt = sum(max(errt))/tn;
    acc = (1 - errt)*100;
    msg = ['test accuracy: ', num2str(acc)];
    disp(msg);

    dn = datetime('now');
    elps = dn - strt;
    
    disp('total runtime: ');
    disp(elps);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%