from .exp_configs.rl import singleagent, multiagent


def get_exp(exp_tag):
    if exp_tag == "singleagent_traffic_light_grid":
        return singleagent.singleagent_traffic_light_grid
    elif exp_tag == "singleagent_ring":
        return singleagent.singleagent_ring
    elif exp_tag == "singleagent_merge_bus_bigger":
        return singleagent.singleagent_merge_bus_bigger
    elif exp_tag == "singleagent_merge_bus":
        return singleagent.singleagent_merge_bus
    elif exp_tag == "singleagent_merge_bus_baseline":
        return singleagent.singleagent_merge_bus_baseline
    elif exp_tag == "singleagent_merge_accel":
        return singleagent.singleagent_merge_accel
    elif exp_tag == "singleagent_merge":
        return singleagent.singleagent_merge
    elif exp_tag == "singleagent_merge_baseline":
        return singleagent.singleagent_merge_baseline
    elif exp_tag == "singleagent_figure_eight":
        return singleagent.singleagent_figure_eight
    elif exp_tag == "singleagent_bottleneck":
        return singleagent.singleagent_bottleneck
    elif exp_tag == "singleagent_bottleneck_baseline":
        return singleagent.singleagent_bottleneck_baseline
    elif exp_tag == "multiagent_traffic_light_grid":
        return multiagent.multiagent_traffic_light_grid
    elif exp_tag == "multiagent_ring":
        return multiagent.multiagent_ring
    elif exp_tag == "multiagent_merge":
        return multiagent.multiagent_merge
    elif exp_tag == "multiagent_i210":
        return multiagent.multiagent_i210
    elif exp_tag == "multiagent_highway":
        return multiagent.multiagent_highway
    elif exp_tag == "multiagent_figure_eight":
        return multiagent.multiagent_figure_eight
    elif exp_tag == "lord_of_the_rings":
        return multiagent.lord_of_the_rings
    elif exp_tag == "adversarial_figure_eight":
        return multiagent.adversarial_figure_eight
    else:
        raise ValueError(exp_tag + " is an invalid scenario!")
