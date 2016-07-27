#!/usr/bin/env bash
set -euf -o pipefail

DIR="$(dirname "$0")"
GA="$DIR/../ga"
GAME="$DIR/hard_nash_game_1.json"

parallel --joblog - <<< "
    # Help tests
    '$GA' --help > /dev/null
    '$GA' help > /dev/null
    '$GA' help nash > /dev/null
    ! '$GA' --fail 2>/dev/null
    ! '$GA' 2>/dev/null

    # Convert test
    '$GA' conv < '$GAME' > /dev/null
    '$GA' conv -i '$GAME' > /dev/null
    '$GA' conv < '$GAME' -o /dev/null
    '$GA' conv -fjson < '$GAME' > /dev/null

    # Dominance test
    '$GA' dom -h > /dev/null
    '$GA' dom -i '$GAME' > /dev/null
    '$GA' dom -cweakdom < '$GAME' -o /dev/null
    '$GA' dom -cstrictdom -s < '$GAME' > /dev/null
    '$GA' dom -cneverbr < '$GAME' > /dev/null

    # Gamegen tests
    ! '$GA' gen 2>/dev/null
    '$GA' gen uzs -n 6 > /dev/null
    ! '$GA' gen ursym 5 -o /dev/null 2>/dev/null
    '$GA' gen ursym 3 4 4 3 > /dev/null
    '$GA' gen noise uniform 1.5 5 -i '$GAME' > /dev/null
    '$GA' gen noise gumbel 1.5 5 < '$GAME' > /dev/null
    '$GA' gen noise bimodal 1.5 5 < '$GAME' > /dev/null
    '$GA' gen noise gaussian 1.5 5 < '$GAME' > /dev/null
    '$GA' gen help > /dev/null
    '$GA' gen help ursym > /dev/null

    # Nash tests
    '$GA' nash < '$GAME' > /dev/null
    '$GA' nash -i '$GAME' -o /dev/null -r1e-2 -d1e-2 -c1e-7 -x100 -s1e-2 -m5 -n -p1
    '$GA' nash -tpure < '$GAME' > /dev/null
    '$GA' nash -tmin-reg-prof < '$GAME' > /dev/null
    '$GA' nash -tmin-reg-grid < '$GAME' > /dev/null
    '$GA' nash -tmin-reg-rand -m10 < '$GAME' > /dev/null
    '$GA' nash -trand -m10 < '$GAME' > /dev/null

    # Payoff tests
    '$GA' pay -i '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 6}, \"hft\": {\"noop\": 1}}]') -o /dev/null
    '$GA' pay < '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 1}, \"hft\": {\"noop\": 1}}]') > /dev/null
    '$GA' pay < '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 6}, \"hft\": {\"noop\": 1}}]') -twelfare > /dev/null
    '$GA' pay < '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 1}, \"hft\": {\"noop\": 1}}]') -twelfare > /dev/null

    # Reduction tests
    '$GA' red background 2 hft 1 < '$GAME' > /dev/null
    '$GA' red -ms 2 1 < '$GAME' > /dev/null
    '$GA' red -thr -s 2 1 -i '$GAME' -o /dev/null
    '$GA' red -ttr < '$GAME' > /dev/null
    '$GA' red -tidr < '$GAME' > /dev/null

    # Regret tests
    '$GA' reg -i '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 6}, \"hft\": {\"noop\": 1}}]') -o /dev/null
    '$GA' reg < '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 1}, \"hft\": {\"noop\": 1}}]') > /dev/null
    '$GA' reg < '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 6}, \"hft\": {\"noop\": 1}}]') -tgains > /dev/null
    '$GA' reg < '$GAME' <(echo '[{\"background\": {\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\": 1}, \"hft\": {\"noop\": 1}}]') -tgains > /dev/null

    # Subgame tests
    '$GA' sub -nd -i '$GAME' -o /dev/null
    '$GA' sub < '$GAME' -f <(echo '[{\"background\": [\"markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9\"], \"hft\": [\"noop\"]}]') > /dev/null
    '$GA' sub -n -t background markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9 hft -s 0 3 4 < '$GAME' > /dev/null
    " | grep -vE $'^([^\t]*\t){6}0'
