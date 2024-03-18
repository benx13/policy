
from policy import policy_noplot, policy_plot



def main(config, source, model_path, show=False, skip_time=0, live=False, device='cpu'):
    if show == True:
        p = policy_plot(config, source, model_path, cap_skip=skip_time, live=live, device=device)
    else:
        p = policy_noplot(config, source, model_path, skip_time, live=live, device=device)
    p.run()


if __name__ == "__main__":
    main('configs/benchmarkcam11_config', 'videos/benchmark_cam_11.mov', 'models/blue_feb_3X+++.mlpackage',show=False, live=False, device='ultralytics')