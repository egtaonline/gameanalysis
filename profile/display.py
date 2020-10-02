"""Aggregate profiling data and generate an rst file"""
import itertools

import tabulate


def update(means, new, count, num=1):
    """Recursively update mean dictionary"""
    for key, val in new.items():
        if isinstance(val, dict):
            update(means.setdefault(key, {}), val, count)
        else:
            value = means.get(key, 0)
            means[key] = value + (val - value) * num / count


def write_file(results, fil):
    """Write file with results"""
    # Compute normalized speeds
    for game in next(iter(results.values())):
        min_speed = min(g[game]["speed"] for g in results.values())
        for games in results.values():
            games[game]["norm_speed"] = games[game]["speed"] / min_speed

    # Aggregate info over all games
    agg_results = {}
    for method, game_info in results.items():
        agg_info = {}
        game_count = 0
        for info in game_info.values():
            count = info.pop("count")
            game_count += count
            update(agg_info, info, game_count, count)
        agg_results[method] = agg_info

    fil.write(
        """.. _profile_nash:

Nash Equilibrium Methods Comparison
===================================

For each method available for Nash equilibrium finding, this lists various
information about the performance across different game types and starting
locations. "Fraction of Eqa" is the mean fraction of all equilibria found via
that method or starting location. "Weigted Fraction (of Eqa)" is the same,
except each equilibrium is down weighted by the number of methods that found
it, thus a larger weighted fraction indicates that this method found more
unique equilibria. "Time" is the average time in seconds it took to run this
method for every starting location. "Normalized Time" sets the minimum time for
each game type and sets it to one, thus somewhat mitigating the fact that
certain games may be more difficult than others. It also provides an easy
comparison metric to for baseline timing.

"""
    )
    fil.write("Comparisons Between Methods\n" "----------------------------------\n\n")
    fil.write(
        tabulate.tabulate(
            sorted(
                (
                    [m.title(), v["card"], v["weight"], v["speed"], v["norm_speed"]]
                    for m, v in agg_results.items()
                ),
                key=lambda x: x[1],
                reverse=True,
            ),
            headers=[
                "Method",
                "Fraction of Eqa",
                "Weighted Fraction",
                "Time (sec)",
                "Normalized Time",
            ],
            tablefmt="rst",
        )
    )
    fil.write("\n\n")

    for method, game_info in results.items():
        title = method.title()
        fil.write(title)
        fil.write("\n")
        fil.writelines(itertools.repeat("-", len(title)))
        fil.write("\n\n")

        agg_info = agg_results[method]
        fil.write("Initial Profile Rates\n" "^^^^^^^^^^^^^^^^^^^^^\n\n")
        fil.write(
            tabulate.tabulate(
                sorted(
                    (
                        [k.capitalize(), v, agg_info["profweight"][k]]
                        for k, v in agg_info["profcard"].items()
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                ),
                headers=["Starting Type", "Fraction of Eqa", "Weighted Fraction"],
                tablefmt="rst",
            )
        )
        fil.write("\n\n")

        fil.write("Compared to Other Methods\n" "^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")
        fil.write(
            tabulate.tabulate(
                sorted(
                    (
                        [
                            m.title(),
                            v,
                            agg_info["norm_speed"] / agg_results[m]["norm_speed"],
                        ]
                        for m, v in agg_info["pair"].items()
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                ),
                headers=["Method", "Fraction of Eqa", "Time Ratio"],
                tablefmt="rst",
            )
        )
        fil.write("\n\n")

        fil.write("By Game Type\n" "^^^^^^^^^^^^\n\n")
        for game, info in game_info.items():
            fil.write(game.capitalize())
            fil.write("\n")
            fil.writelines(itertools.repeat('"', len(game)))
            fil.write("\n\n")
            fil.write(
                tabulate.tabulate(
                    [
                        ["Fraction of Eqa", info["card"]],
                        ["Weighted Fraction of Eqa", info["weight"]],
                        ["Time (sec)", info["speed"]],
                        ["Normalized Time", info["norm_speed"]],
                    ],
                    headers=["Metric", "Value"],
                    tablefmt="rst",
                )
            )
            fil.write("\n\n")
