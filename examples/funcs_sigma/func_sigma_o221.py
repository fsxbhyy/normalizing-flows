def graphfunc(leaf):
    g18745 = leaf[:, 0]
    g18752 = leaf[:, 1]
    g18842 = leaf[:, 2]
    g18860 = (g18752 * g18842 * -1.0)
    g18882 = leaf[:, 3]
    g18916 = (g18860 * g18882)
    g18838 = leaf[:, 4]
    g18870 = (g18838 * g18842 * -1.0)
    g18881 = leaf[:, 5]
    g18930 = (g18870 * g18881)
    g18931 = (g18916 + g18930)
    g18839 = leaf[:, 6]
    g18865 = (g18839 * g18842 * -1.0)
    g18759 = leaf[:, 7]
    g18947 = (g18865 * g18759)
    g18948 = (g18931 + g18947)
    g19076 = (g18745 * g18948)
    g18835 = leaf[:, 8]
    g18915 = (g18860 * g18759)
    g19095 = (g18835 * g18915)
    g19096 = (g19076 + g19095)
    g18834 = leaf[:, 9]
    g18917 = (g18860 * g18881)
    g18926 = (g18870 * g18759)
    g18927 = (g18917 + g18926)
    g19129 = (g18834 * g18927)
    g19130 = (g19096 + g19129)
    g18739 = leaf[:, 10]
    g19262 = (g19130 * g18739 * -1.0)
    g18746 = leaf[:, 11]
    g18856 = (g18752 * g18746 * -1.0)
    g18921 = (g18856 * g18882)
    g18866 = (g18838 * g18746 * -1.0)
    g18938 = (g18866 * g18881)
    g18939 = (g18921 + g18938)
    g18861 = (g18839 * g18746 * -1.0)
    g18961 = (g18861 * g18759)
    g18962 = (g18939 + g18961)
    g19078 = (g18745 * g18962)
    g18920 = (g18856 * g18759)
    g19097 = (g18835 * g18920)
    g19098 = (g19078 + g19097)
    g18922 = (g18856 * g18881)
    g18934 = (g18866 * g18759)
    g18935 = (g18922 + g18934)
    g19131 = (g18834 * g18935)
    g19132 = (g19098 + g19131)
    g19195 = leaf[:, 12]
    g19277 = (g19132 * g19195 * -1.0)
    g19278 = (g19262 + g19277)
    g19360 = (g19278)
    root0 = g19360
    g18785 = leaf[:, 13]
    g19393 = (g18739 * -1.0 + g18785)
    g19379 = leaf[:, 14]
    g19412 = (g19195 * -1.0 + g19379)
    g19427 = (g19393 * g19412)
    g19397 = (g19195 * -1.0 + g19379)
    g19408 = (g18739 * -1.0 + g18785)
    g19440 = (g19397 * g19408)
    g19441 = (g19427 + g19440)
    g19462 = (g18739 * -1.0 * g19195 * -1.0)
    g19475 = (g19195 * -1.0 * g18739 * -1.0)
    g19476 = (g19462 + g19475)
    g19497 = (g19441 * -1.0 + g19476 * -1.0)
    g18817 = leaf[:, 15]
    g19536 = (g19497 * g18817)
    g19503 = leaf[:, 16]
    g19547 = (g19536 * g19503)
    g19498 = leaf[:, 17]
    g19538 = (g19497 * g19498)
    g19502 = leaf[:, 18]
    g19561 = (g19538 * g19502)
    g19562 = (g19547 + g19561)
    g19499 = leaf[:, 19]
    g19537 = (g19497 * g19499)
    g18818 = leaf[:, 20]
    g19578 = (g19537 * g18818)
    g19579 = (g19562 + g19578)
    g19707 = (g18745 * g19579)
    g19546 = (g19536 * g18818)
    g19726 = (g18835 * g19546)
    g19727 = (g19707 + g19726)
    g19548 = (g19536 * g19502)
    g19557 = (g19538 * g18818)
    g19558 = (g19548 + g19557)
    g19760 = (g18834 * g19558)
    g19761 = (g19727 + g19760)
    g19840 = (g19393 * g19195 * -1.0)
    g19853 = (g19397 * g18739 * -1.0)
    g19854 = (g19840 + g19853)
    g19875 = (g18739 * -1.0 * g19412)
    g19888 = (g19195 * -1.0 * g19408)
    g19889 = (g19875 + g19888)
    g19910 = (g19854 * -1.0 + g19889 * -1.0)
    g19941 = (g19910 * g18817)
    g19952 = (g19941 * g19503)
    g19943 = (g19910 * g19498)
    g19966 = (g19943 * g19502)
    g19967 = (g19952 + g19966)
    g19942 = (g19910 * g19499)
    g19983 = (g19942 * g18818)
    g19984 = (g19967 + g19983)
    g20112 = (g18745 * g19984)
    g19951 = (g19941 * g18818)
    g20131 = (g18835 * g19951)
    g20132 = (g20112 + g20131)
    g19953 = (g19941 * g19502)
    g19962 = (g19943 * g18818)
    g19963 = (g19953 + g19962)
    g20165 = (g18834 * g19963)
    g20166 = (g20132 + g20165)
    g20287 = (g19761 + g20166 * -0.5)
    root1 = g20287
    return root0,root1
