def graphfunc(leaf):
    g18719 = leaf[:, 0]
    g18726 = leaf[:, 1]
    g18720 = leaf[:, 2]
    g18825 = (g18726 * g18720 * -1.0)
    g18733 = leaf[:, 3]
    g18864 = (g18825 * g18733)
    g18948 = (g18719 * g18864)
    g19013 = leaf[:, 4]
    g19036 = (g18948 * g19013 * -1.0)
    g18815 = leaf[:, 5]
    g18826 = (g18726 * g18815 * -1.0)
    g18903 = (g18826 * g18733)
    g18958 = (g18719 * g18903)
    g18713 = leaf[:, 6]
    g19075 = (g18958 * g18713 * -1.0)
    g19076 = (g19036 + g19075)
    g18814 = leaf[:, 7]
    g18828 = (g18726 * g18814 * -1.0)
    g18921 = (g18828 * g18733)
    g18963 = (g18719 * g18921)
    g19012 = leaf[:, 8]
    g19097 = (g18963 * g19012 * -1.0)
    g19098 = (g19076 + g19097)
    g19109 = (g19098)
    root0 = g19109
    g18759 = leaf[:, 9]
    g19126 = (g18713 * -1.0 + g18759)
    g19116 = leaf[:, 10]
    g19139 = (g19013 * -1.0 + g19116)
    g19151 = (g19126 * g19139)
    g19127 = (g19013 * -1.0 + g19116)
    g19138 = (g18713 * -1.0 + g18759)
    g19154 = (g19127 * g19138)
    g19155 = (g19151 + g19154)
    g19115 = leaf[:, 11]
    g19129 = (g19012 * -1.0 + g19115)
    g19141 = (g19012 * -1.0 + g19115)
    g19164 = (g19129 * g19141)
    g19165 = (g19155 + g19164)
    g19175 = (g18713 * -1.0 * g19013 * -1.0)
    g19178 = (g19013 * -1.0 * g18713 * -1.0)
    g19179 = (g19175 + g19178)
    g19188 = ((g19012)**2)
    g19189 = (g19179 + g19188)
    g19199 = (g19165 * -1.0 + g19189 * -1.0)
    g18791 = leaf[:, 12]
    g19220 = (g19199 * g18791)
    g18792 = leaf[:, 13]
    g19275 = (g19220 * g18792)
    g19330 = (g18719 * g19275)
    g19393 = (g19126 * g19013 * -1.0)
    g19396 = (g19127 * g18713 * -1.0)
    g19397 = (g19393 + g19396)
    g19406 = (g19129 * g19012 * -1.0)
    g19407 = (g19397 + g19406)
    g19417 = (g18713 * -1.0 * g19139)
    g19420 = (g19013 * -1.0 * g19138)
    g19421 = (g19417 + g19420)
    g19430 = (g19012 * -1.0 * g19141)
    g19431 = (g19421 + g19430)
    g19441 = (g19407 * -1.0 + g19431 * -1.0)
    g19456 = (g19441 * g18791)
    g19511 = (g19456 * g18792)
    g19566 = (g18719 * g19511)
    g19662 = (g19330 + g19566 * -0.5)
    root1 = g19662
    return root0,root1

