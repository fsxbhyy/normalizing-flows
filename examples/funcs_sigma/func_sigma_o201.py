def graphfunc(leaf):
    g18679 = leaf[:, 0]
    g18686 = leaf[:, 1]
    g18680 = leaf[:, 2]
    g18775 = (g18686 * g18680 * -1.0)
    g18693 = leaf[:, 3]
    g18786 = (g18775 * g18693)
    g18800 = (g18679 * g18786)
    g18808 = leaf[:, 4]
    g18816 = (g18800 * g18808 * -1.0)
    g18770 = leaf[:, 5]
    g18776 = (g18686 * g18770 * -1.0)
    g18791 = (g18776 * g18693)
    g18803 = (g18679 * g18791)
    g18673 = leaf[:, 6]
    g18821 = (g18803 * g18673 * -1.0)
    g18822 = (g18816 + g18821)
    g18826 = (g18822)
    root0 = g18826
    g18719 = leaf[:, 7]
    g18832 = (g18673 * -1.0 + g18719)
    g18827 = leaf[:, 8]
    g18839 = (g18808 * -1.0 + g18827)
    g18845 = (g18832 * g18839)
    g18833 = (g18808 * -1.0 + g18827)
    g18838 = (g18673 * -1.0 + g18719)
    g18846 = (g18833 * g18838)
    g18847 = (g18845 + g18846)
    g18853 = (g18673 * -1.0 * g18808 * -1.0)
    g18854 = (g18808 * -1.0 * g18673 * -1.0)
    g18855 = (g18853 + g18854)
    g18861 = (g18847 * -1.0 + g18855 * -1.0)
    g18755 = leaf[:, 9]
    g18870 = (g18861 * g18755)
    g18756 = leaf[:, 10]
    g18879 = (g18870 * g18756)
    g18891 = (g18679 * g18879)
    g18901 = (g18832 * g18808 * -1.0)
    g18902 = (g18833 * g18673 * -1.0)
    g18903 = (g18901 + g18902)
    g18909 = (g18673 * -1.0 * g18839)
    g18910 = (g18808 * -1.0 * g18838)
    g18911 = (g18909 + g18910)
    g18917 = (g18903 * -1.0 + g18911 * -1.0)
    g18924 = (g18917 * g18755)
    g18933 = (g18924 * g18756)
    g18945 = (g18679 * g18933)
    g18961 = (g18891 + g18945 * -0.5)
    root1 = g18961
    return root0,root1
