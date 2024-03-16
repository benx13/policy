
from policy import policy_noplot, policy_plot



def main(config, source, show=False, skip_time=0, live=False):
    if show == True:
        p = policy_plot(config, source, cap_skip=skip_time, live=live)
    else:
        p = policy_noplot(config, source, skip_time, live=live)
    p.run()


if __name__ == "__main__":
    main('configs/benchmarkcam14_config', 'videos/benchmark_cam_14.mov', show=True, live=True)