def graphfunc(leaf):
    g18719 = leaf[:, 0]
    g18811 = leaf[:, 1]
    g18814 = leaf[:, 2]
    g18836 = (g18811 * g18814 * -1.0)
    g18733 = leaf[:, 3]
    g18868 = (g18836 * g18733)
    g18726 = leaf[:, 4]
    g18828 = (g18726 * g18814 * -1.0)
    g18841 = leaf[:, 5]
    g18924 = (g18828 * g18841)
    g18925 = (g18868 + g18924)
    g18949 = (g18719 * g18925)
    g18808 = leaf[:, 6]
    g18921 = (g18828 * g18733)
    g19002 = (g18808 * g18921)
    g19003 = (g18949 + g19002)
    g18713 = leaf[:, 7]
    g19039 = (g19003 * g18713 * -1.0)
    g18720 = leaf[:, 8]
    g18825 = (g18726 * g18720 * -1.0)
    g18866 = (g18825 * g18841)
    g18833 = (g18811 * g18720 * -1.0)
    g18871 = (g18833 * g18733)
    g18872 = (g18866 + g18871)
    g18950 = (g18719 * g18872)
    g18864 = (g18825 * g18733)
    g18980 = (g18808 * g18864)
    g18981 = (g18950 + g18980)
    g19012 = leaf[:, 9]
    g19047 = (g18981 * g19012 * -1.0)
    g19048 = (g19039 + g19047)
    g19100 = (g19048)
    root0 = g19100
    g18759 = leaf[:, 10]
    g19126 = (g18713 * -1.0 + g18759)
    g19115 = leaf[:, 11]
    g19141 = (g19012 * -1.0 + g19115)
    g19153 = (g19126 * g19141)
    g19129 = (g19012 * -1.0 + g19115)
    g19138 = (g18713 * -1.0 + g18759)
    g19160 = (g19129 * g19138)
    g19161 = (g19153 + g19160)
    g19177 = (g18713 * -1.0 * g19012 * -1.0)
    g19184 = (g19012 * -1.0 * g18713 * -1.0)
    g19185 = (g19177 + g19184)
    g19201 = (g19161 * -1.0 + g19185 * -1.0)
    g19202 = leaf[:, 12]
    g19230 = (g19201 * g19202)
    g18792 = leaf[:, 13]
    g19240 = (g19230 * g18792)
    g18791 = leaf[:, 14]
    g19228 = (g19201 * g18791)
    g19205 = leaf[:, 15]
    g19296 = (g19228 * g19205)
    g19297 = (g19240 + g19296)
    g19321 = (g18719 * g19297)
    g19293 = (g19228 * g18792)
    g19374 = (g18808 * g19293)
    g19375 = (g19321 + g19374)
    g19395 = (g19126 * g19012 * -1.0)
    g19402 = (g19129 * g18713 * -1.0)
    g19403 = (g19395 + g19402)
    g19419 = (g18713 * -1.0 * g19141)
    g19426 = (g19012 * -1.0 * g19138)
    g19427 = (g19419 + g19426)
    g19443 = (g19403 * -1.0 + g19427 * -1.0)
    g19466 = (g19443 * g19202)
    g19476 = (g19466 * g18792)
    g19464 = (g19443 * g18791)
    g19532 = (g19464 * g19205)
    g19533 = (g19476 + g19532)
    g19557 = (g18719 * g19533)
    g19529 = (g19464 * g18792)
    g19610 = (g18808 * g19529)
    g19611 = (g19557 + g19610)
    g19653 = (g19375 + g19611 * -0.5)
    root1 = g19653
    return root0,root1

