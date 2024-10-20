import stim
import numpy as np

from split_mwpm import SplitMatching


def test_SplitMatching():
    dem = stim.DetectorErrorModel(
        """
        error(0.001) D0 D3
    error(0.001998) D0 D29
    error(0.001) D0 L0
    error(0.001) D1 D3
    error(0.001) D1 D4
    error(0.001998) D1 D30
    error(0.001) D2
    error(0.001) D2 D4
    error(0.001998) D2 D31
    error(0.001) D3 D5
    error(0.001) D3 D6
    error(0.001998) D3 D32
    error(0.001) D4 D6
    error(0.001) D4 D7
    error(0.001998) D4 D33
    error(0.001) D5 D8
    error(0.001998) D5 D34
    error(0.001998) D5 L0
    error(0.001) D6 D8
    error(0.001) D6 D9
    error(0.001998) D6 D35
    error(0.001998) D7
    error(0.001) D7 D9
    error(0.001998) D7 D36
    error(0.001) D8 D10
    error(0.001) D8 D11
    error(0.001998) D8 D37
    error(0.001) D9 D11
    error(0.001) D9 D12
    error(0.001998) D9 D38
    error(0.001) D10 D13
    error(0.001998) D10 D39
    error(0.001998) D10 L0
    error(0.001) D11 D13
    error(0.001) D11 D14
    error(0.001998) D11 D40
    error(0.001998) D12
    error(0.001) D12 D14
    error(0.001998) D12 D41
    error(0.001998) D13 D42
    error(0.001998) D14 D43
    error(0.001998) D29 D32
    error(0.001998) D29 D58
    error(0.001998) D29 L0
    error(0.001998) D30 D32
    error(0.001) D30 D32 D44 D47
    error(0.001998) D30 D33
    error(0.001) D30 D33 D47 D51
    error(0.001998) D30 D47 D76
    error(0.001998) D30 D59
    error(0.001998) D31
    error(0.001998) D31 D33
    error(0.001) D31 D33 D51 D54
    error(0.001) D31 D54
    error(0.001998) D31 D54 D83
    error(0.001998) D31 D60
    error(0.001998) D32 D34
    error(0.001998) D32 D35
    error(0.001) D32 D35 D44 D48
    error(0.001998) D32 D44 D73
    error(0.001998) D32 D44 L0
    error(0.001998) D32 D61
    error(0.001998) D33 D35
    error(0.001) D33 D35 D48 D51
    error(0.001998) D33 D36
    error(0.001) D33 D36 D51 D55
    error(0.001998) D33 D51 D80
    error(0.001998) D33 D62
    error(0.001) D34 D35 D45 D48
    error(0.001998) D34 D37
    error(0.001) D34 D37 D45 D49
    error(0.001998) D34 D45 D74
    error(0.001998) D34 D45 L0
    error(0.001998) D34 D63
    error(0.00398802) D34 L0
    error(0.001998) D35 D37
    error(0.001998) D35 D38
    error(0.001) D35 D38 D48 D52
    error(0.001998) D35 D48 D77
    error(0.001998) D35 D64
    error(0.00398802) D36
    error(0.001998) D36 D38
    error(0.001) D36 D38 D52 D55
    error(0.001998) D36 D55
    error(0.001998) D36 D55 D84
    error(0.001998) D36 D65
    error(0.001) D37 D38 D49 D52
    error(0.001998) D37 D39
    error(0.001) D37 D39 D46 D49
    error(0.001998) D37 D40
    error(0.001) D37 D40 D49 D53
    error(0.001998) D37 D49 D78
    error(0.001998) D37 D66
    error(0.001998) D38 D40
    error(0.001998) D38 D41
    error(0.001) D38 D41 D52 D56
    error(0.001998) D38 D52 D81
    error(0.001998) D38 D67
    error(0.001998) D39 D42
    error(0.001) D39 D42 D46 D50
    error(0.001998) D39 D46 D75
    error(0.001998) D39 D46 L0
    error(0.001998) D39 D68
    error(0.00398802) D39 L0
    error(0.001) D40 D41 D53 D56
    error(0.001998) D40 D42
    error(0.001) D40 D42 D50 D53
    error(0.001998) D40 D43
    error(0.001) D40 D43 D53 D57
    error(0.001998) D40 D53 D82
    error(0.001998) D40 D69
    error(0.00398802) D41
    error(0.001998) D41 D43
    error(0.001998) D41 D56
    error(0.001998) D41 D56 D85
    error(0.001998) D41 D70
    error(0.001998) D42 D50 D79
    error(0.001998) D42 D71
    error(0.001) D43 D57
    error(0.001998) D43 D57 D86
    error(0.001998) D43 D72
    error(0.00398802) D44
    error(0.001) D44 D47
    error(0.001) D44 D48
    error(0.00398802) D45
    error(0.001) D45 D48
    error(0.001) D45 D49
    error(0.00398802) D46
    error(0.001) D46 D49
    error(0.001) D46 D50
    error(0.001998) D47
    error(0.001) D47 D51
    error(0.001998) D48
    error(0.001) D48 D51
    error(0.001) D48 D52
    error(0.001998) D49
    error(0.001) D49 D52
    error(0.001) D49 D53
    error(0.001998) D50
    error(0.001) D50 D53
    error(0.001998) D51
    error(0.001) D51 D54
    error(0.001) D51 D55
    error(0.001998) D52
    error(0.001) D52 D55
    error(0.001) D52 D56
    error(0.001998) D53
    error(0.001) D53 D56
    error(0.001) D53 D57
    error(0.002994) D54
    error(0.00398802) D55
    error(0.00398802) D56
    error(0.002994) D57
    error(0.001) D58 D61
    error(0.001998) D58 D87
    error(0.001) D58 L0
    error(0.001) D59 D61
    error(0.001) D59 D62
    error(0.001998) D59 D88
    error(0.001) D60
    error(0.001) D60 D62
    error(0.001998) D60 D89
    error(0.001) D61 D63
    error(0.001) D61 D64
    error(0.001998) D61 D90
    error(0.001) D62 D64
    error(0.001) D62 D65
    error(0.001998) D62 D91
    error(0.001) D63 D66
    error(0.001998) D63 D92
    error(0.001998) D63 L0
    error(0.001) D64 D66
    error(0.001) D64 D67
    error(0.001998) D64 D93
    error(0.001998) D65
    error(0.001) D65 D67
    error(0.001998) D65 D94
    error(0.001) D66 D68
    error(0.001) D66 D69
    error(0.001998) D66 D95
    error(0.001) D67 D69
    error(0.001) D67 D70
    error(0.001998) D67 D96
    error(0.001) D68 D71
    error(0.001998) D68 D97
    error(0.001998) D68 L0
    error(0.001) D69 D71
    error(0.001) D69 D72
    error(0.001998) D69 D98
    error(0.001998) D70
    error(0.001) D70 D72
    error(0.001998) D70 D99
    error(0.001998) D71 D100
    error(0.001998) D72 D101
    error(0.001) D73 D76
    error(0.001) D73 D77
    error(0.001998) D73 D102
    error(0.001998) D73 L0
    error(0.001) D74 D77
    error(0.001) D74 D78
    error(0.001998) D74 D103
    error(0.001998) D74 L0
    error(0.001) D75 D78
    error(0.001) D75 D79
    error(0.001998) D75 D104
    error(0.001998) D75 L0
    error(0.001) D76 D80
    error(0.001998) D76 D105
    error(0.001) D77 D80
    error(0.001) D77 D81
    error(0.001998) D77 D106
    error(0.001) D78 D81
    error(0.001) D78 D82
    error(0.001998) D78 D107
    error(0.001) D79 D82
    error(0.001998) D79 D108
    error(0.001) D80 D83
    error(0.001) D80 D84
    error(0.001998) D80 D109
    error(0.001) D81 D84
    error(0.001) D81 D85
    error(0.001998) D81 D110
    error(0.001) D82 D85
    error(0.001) D82 D86
    error(0.001998) D82 D111
    error(0.001) D83
    error(0.001998) D83 D112
    error(0.001998) D84
    error(0.001998) D84 D113
    error(0.001998) D85
    error(0.001998) D85 D114
    error(0.001) D86
    error(0.001998) D86 D115
    error(0.001998) D87 D90
    error(0.001998) D87 D116
    error(0.001998) D87 L0
    error(0.001998) D88 D90
    error(0.001) D88 D90 D102 D105
    error(0.001998) D88 D91
    error(0.001) D88 D91 D105 D109
    error(0.001998) D88 D105 D134
    error(0.001998) D88 D117
    error(0.001998) D89
    error(0.001998) D89 D91
    error(0.001) D89 D91 D109 D112
    error(0.001) D89 D112
    error(0.001998) D89 D112 D141
    error(0.001998) D89 D118
    error(0.001998) D90 D92
    error(0.001998) D90 D93
    error(0.001) D90 D93 D102 D106
    error(0.001998) D90 D102
    error(0.001998) D90 D102 D131
    error(0.001998) D90 D119
    error(0.001998) D91 D93
    error(0.001) D91 D93 D106 D109
    error(0.001998) D91 D94
    error(0.001) D91 D94 D109 D113
    error(0.001998) D91 D109 D138
    error(0.001998) D91 D120
    error(0.001) D92 D93 D103 D106
    error(0.001998) D92 D95
    error(0.001) D92 D95 D103 D107
    error(0.001998) D92 D103
    error(0.001998) D92 D103 D132
    error(0.001998) D92 D121
    error(0.00398802) D92 L0
    error(0.001998) D93 D95
    error(0.001998) D93 D96
    error(0.001) D93 D96 D106 D110
    error(0.001998) D93 D106 D135
    error(0.001998) D93 D122
    error(0.00398802) D94
    error(0.001998) D94 D96
    error(0.001) D94 D96 D110 D113
    error(0.001998) D94 D113
    error(0.001998) D94 D113 D142
    error(0.001998) D94 D123
    error(0.001) D95 D96 D107 D110
    error(0.001998) D95 D97
    error(0.001) D95 D97 D104 D107
    error(0.001998) D95 D98
    error(0.001) D95 D98 D107 D111
    error(0.001998) D95 D107 D136
    error(0.001998) D95 D124
    error(0.001998) D96 D98
    error(0.001998) D96 D99
    error(0.001) D96 D99 D110 D114
    error(0.001998) D96 D110 D139
    error(0.001998) D96 D125
    error(0.001998) D97 D100
    error(0.001) D97 D100 D104 D108
    error(0.001998) D97 D104
    error(0.001998) D97 D104 D133
    error(0.001998) D97 D126
    error(0.00398802) D97 L0
    error(0.001) D98 D99 D111 D114
    error(0.001998) D98 D100
    error(0.001) D98 D100 D108 D111
    error(0.001998) D98 D101
    error(0.001) D98 D101 D111 D115
    error(0.001998) D98 D111 D140
    error(0.001998) D98 D127
    error(0.00398802) D99
    error(0.001998) D99 D101
    error(0.001998) D99 D114
    error(0.001998) D99 D114 D143
    error(0.001998) D99 D128
    error(0.001998) D100 D108 D137
    error(0.001998) D100 D129
    error(0.001) D101 D115
    error(0.001998) D101 D115 D144
    error(0.001998) D101 D130
    error(0.001) D102 D105
    error(0.001) D102 D106
    error(0.001998) D102 L0
    error(0.001) D103 D106
    error(0.001) D103 D107
    error(0.001998) D103 L0
    error(0.001) D104 D107
    error(0.001) D104 D108
    error(0.001998) D104 L0
    error(0.001) D105 D109
    error(0.001) D106 D109
    error(0.001) D106 D110
    error(0.001) D107 D110
    error(0.001) D107 D111
    error(0.001) D108 D111
    error(0.001) D109 D112
    error(0.001) D109 D113
    error(0.001) D110 D113
    error(0.001) D110 D114
    error(0.001) D111 D114
    error(0.001) D111 D115
    error(0.001) D112
    error(0.001998) D113
    error(0.001998) D114
    error(0.001) D115
    error(0.001) D116 D119
    error(0.001998) D116 D145
    error(0.001) D116 L0
    error(0.001) D117 D119
    error(0.001) D117 D120
    error(0.001998) D117 D146
    error(0.001) D118
    error(0.001) D118 D120
    error(0.001998) D118 D147
    error(0.001) D119 D121
    error(0.001) D119 D122
    error(0.001998) D119 D148
    error(0.001) D120 D122
    error(0.001) D120 D123
    error(0.001998) D120 D149
    error(0.001) D121 D124
    error(0.001998) D121 D150
    error(0.001998) D121 L0
    error(0.001) D122 D124
    error(0.001) D122 D125
    error(0.001998) D122 D151
    error(0.001998) D123
    error(0.001) D123 D125
    error(0.001998) D123 D152
    error(0.001) D124 D126
    error(0.001) D124 D127
    error(0.001998) D124 D153
    error(0.001) D125 D127
    error(0.001) D125 D128
    error(0.001998) D125 D154
    error(0.001) D126 D129
    error(0.001998) D126 D155
    error(0.001998) D126 L0
    error(0.001) D127 D129
    error(0.001) D127 D130
    error(0.001998) D127 D156
    error(0.001998) D128
    error(0.001) D128 D130
    error(0.001998) D128 D157
    error(0.001998) D129 D158
    error(0.001998) D130 D159
    error(0.00398802) D131
    error(0.001) D131 D134
    error(0.001) D131 D135
    error(0.00398802) D132
    error(0.001) D132 D135
    error(0.001) D132 D136
    error(0.00398802) D133
    error(0.001) D133 D136
    error(0.001) D133 D137
    error(0.001998) D134
    error(0.001) D134 D138
    error(0.001998) D135
    error(0.001) D135 D138
    error(0.001) D135 D139
    error(0.001998) D136
    error(0.001) D136 D139
    error(0.001) D136 D140
    error(0.001998) D137
    error(0.001) D137 D140
    error(0.001998) D138
    error(0.001) D138 D141
    error(0.001) D138 D142
    error(0.001998) D139
    error(0.001) D139 D142
    error(0.001) D139 D143
    error(0.001998) D140
    error(0.001) D140 D143
    error(0.001) D140 D144
    error(0.002994) D141
    error(0.00398802) D142
    error(0.00398802) D143
    error(0.002994) D144
    error(0.002994) D145 D148
    error(0.002994) D145 L0
    error(0.002994) D146 D148
    error(0.002994) D146 D149
    error(0.002994) D147
    error(0.002994) D147 D149
    error(0.002994) D148 D150
    error(0.002994) D148 D151
    error(0.002994) D149 D151
    error(0.002994) D149 D152
    error(0.002994) D150 D153
    error(0.00597008) D150 L0
    error(0.002994) D151 D153
    error(0.002994) D151 D154
    error(0.00597008) D152
    error(0.002994) D152 D154
    error(0.002994) D153 D155
    error(0.002994) D153 D156
    error(0.002994) D154 D156
    error(0.002994) D154 D157
    error(0.002994) D155 D158
    error(0.00597008) D155 L0
    error(0.002994) D156 D158
    error(0.002994) D156 D159
    error(0.00597008) D157
    error(0.002994) D157 D159
    detector D15
    detector D16
    detector D17
    detector D18
    detector D19
    detector D20
    detector D21
    detector D22
    detector D23
    detector D24
    detector D25
    detector D26
    detector D27
    detector D28
    """
    )

    dem_z = stim.DetectorErrorModel(
        """
        error(0.00398802) D44
    error(0.001998) D44 D47
    error(0.001998) D44 D48
    error(0.001998) D44 D73
    error(0.001998) D44 L0
    error(0.00398802) D45
    error(0.001998) D45 D48
    error(0.001998) D45 D49
    error(0.001998) D45 D74
    error(0.001998) D45 L0
    error(0.00398802) D46
    error(0.001998) D46 D49
    error(0.001998) D46 D50
    error(0.001998) D46 D75
    error(0.001998) D46 L0
    error(0.001998) D47
    error(0.001998) D47 D51
    error(0.001998) D47 D76
    error(0.001998) D48
    error(0.001998) D48 D51
    error(0.001998) D48 D52
    error(0.001998) D48 D77
    error(0.001998) D49
    error(0.001998) D49 D52
    error(0.001998) D49 D53
    error(0.001998) D49 D78
    error(0.001998) D50
    error(0.001998) D50 D53
    error(0.001998) D50 D79
    error(0.001998) D51
    error(0.001998) D51 D54
    error(0.001998) D51 D55
    error(0.001998) D51 D80
    error(0.001998) D52
    error(0.001998) D52 D55
    error(0.001998) D52 D56
    error(0.001998) D52 D81
    error(0.001998) D53
    error(0.001998) D53 D56
    error(0.001998) D53 D57
    error(0.001998) D53 D82
    error(0.00398802) D54
    error(0.001998) D54 D83
    error(0.00597008) D55
    error(0.001998) D55 D84
    error(0.00597008) D56
    error(0.001998) D56 D85
    error(0.00398802) D57
    error(0.001998) D57 D86
    error(0.001) D73 D76
    error(0.001) D73 D77
    error(0.001998) D73 D102
    error(0.001998) D73 L0
    error(0.001) D74 D77
    error(0.001) D74 D78
    error(0.001998) D74 D103
    error(0.001998) D74 L0
    error(0.001) D75 D78
    error(0.001) D75 D79
    error(0.001998) D75 D104
    error(0.001998) D75 L0
    error(0.001) D76 D80
    error(0.001998) D76 D105
    error(0.001) D77 D80
    error(0.001) D77 D81
    error(0.001998) D77 D106
    error(0.001) D78 D81
    error(0.001) D78 D82
    error(0.001998) D78 D107
    error(0.001) D79 D82
    error(0.001998) D79 D108
    error(0.001) D80 D83
    error(0.001) D80 D84
    error(0.001998) D80 D109
    error(0.001) D81 D84
    error(0.001) D81 D85
    error(0.001998) D81 D110
    error(0.001) D82 D85
    error(0.001) D82 D86
    error(0.001998) D82 D111
    error(0.001) D83
    error(0.001998) D83 D112
    error(0.001998) D84
    error(0.001998) D84 D113
    error(0.001998) D85
    error(0.001998) D85 D114
    error(0.001) D86
    error(0.001998) D86 D115
    error(0.001998) D102
    error(0.001998) D102 D105
    error(0.001998) D102 D106
    error(0.001998) D102 D131
    error(0.001998) D102 L0
    error(0.001998) D103
    error(0.001998) D103 D106
    error(0.001998) D103 D107
    error(0.001998) D103 D132
    error(0.001998) D103 L0
    error(0.001998) D104
    error(0.001998) D104 D107
    error(0.001998) D104 D108
    error(0.001998) D104 D133
    error(0.001998) D104 L0
    error(0.001998) D105 D109
    error(0.001998) D105 D134
    error(0.001998) D106 D109
    error(0.001998) D106 D110
    error(0.001998) D106 D135
    error(0.001998) D107 D110
    error(0.001998) D107 D111
    error(0.001998) D107 D136
    error(0.001998) D108 D111
    error(0.001998) D108 D137
    error(0.001998) D109 D112
    error(0.001998) D109 D113
    error(0.001998) D109 D138
    error(0.001998) D110 D113
    error(0.001998) D110 D114
    error(0.001998) D110 D139
    error(0.001998) D111 D114
    error(0.001998) D111 D115
    error(0.001998) D111 D140
    error(0.001998) D112
    error(0.001998) D112 D141
    error(0.00398802) D113
    error(0.001998) D113 D142
    error(0.00398802) D114
    error(0.001998) D114 D143
    error(0.001998) D115
    error(0.001998) D115 D144
    error(0.00398802) D131
    error(0.001) D131 D134
    error(0.001) D131 D135
    error(0.00398802) D132
    error(0.001) D132 D135
    error(0.001) D132 D136
    error(0.00398802) D133
    error(0.001) D133 D136
    error(0.001) D133 D137
    error(0.001998) D134
    error(0.001) D134 D138
    error(0.001998) D135
    error(0.001) D135 D138
    error(0.001) D135 D139
    error(0.001998) D136
    error(0.001) D136 D139
    error(0.001) D136 D140
    error(0.001998) D137
    error(0.001) D137 D140
    error(0.001998) D138
    error(0.001) D138 D141
    error(0.001) D138 D142
    error(0.001998) D139
    error(0.001) D139 D142
    error(0.001) D139 D143
    error(0.001998) D140
    error(0.001) D140 D143
    error(0.001) D140 D144
    error(0.002994) D141
    error(0.00398802) D142
    error(0.00398802) D143
    error(0.002994) D144
    error(0.0476266) L0
    detector D0
    detector D1
    detector D2
    detector D3
    detector D4
    detector D5
    detector D6
    detector D7
    detector D8
    detector D9
    detector D10
    detector D11
    detector D12
    detector D13
    detector D14
    detector D15
    detector D16
    detector D17
    detector D18
    detector D19
    detector D20
    detector D21
    detector D22
    detector D23
    detector D24
    detector D25
    detector D26
    detector D27
    detector D28
    detector D29
    detector D30
    detector D31
    detector D32
    detector D33
    detector D34
    detector D35
    detector D36
    detector D37
    detector D38
    detector D39
    detector D40
    detector D41
    detector D42
    detector D43
    detector D58
    detector D59
    detector D60
    detector D61
    detector D62
    detector D63
    detector D64
    detector D65
    detector D66
    detector D67
    detector D68
    detector D69
    detector D70
    detector D71
    detector D72
    detector D87
    detector D88
    detector D89
    detector D90
    detector D91
    detector D92
    detector D93
    detector D94
    detector D95
    detector D96
    detector D97
    detector D98
    detector D99
    detector D100
    detector D101
    detector D116
    detector D117
    detector D118
    detector D119
    detector D120
    detector D121
    detector D122
    detector D123
    detector D124
    detector D125
    detector D126
    detector D127
    detector D128
    detector D129
    detector D130
    detector D145
    detector D146
    detector D147
    detector D148
    detector D149
    detector D150
    detector D151
    detector D152
    detector D153
    detector D154
    detector D155
    detector D156
    detector D157
    detector D158
    detector D159
    """
    )

    dem_x = stim.DetectorErrorModel(
        """
        error(0.001) D0 D3
    error(0.001998) D0 D29
    error(0.001) D0 L0
    error(0.001) D1 D3
    error(0.001) D1 D4
    error(0.001998) D1 D30
    error(0.001) D2
    error(0.001) D2 D4
    error(0.001998) D2 D31
    error(0.001) D3 D5
    error(0.001) D3 D6
    error(0.001998) D3 D32
    error(0.001) D4 D6
    error(0.001) D4 D7
    error(0.001998) D4 D33
    error(0.001) D5 D8
    error(0.001998) D5 D34
    error(0.001998) D5 L0
    error(0.001) D6 D8
    error(0.001) D6 D9
    error(0.001998) D6 D35
    error(0.001998) D7
    error(0.001) D7 D9
    error(0.001998) D7 D36
    error(0.001) D8 D10
    error(0.001) D8 D11
    error(0.001998) D8 D37
    error(0.001) D9 D11
    error(0.001) D9 D12
    error(0.001998) D9 D38
    error(0.001) D10 D13
    error(0.001998) D10 D39
    error(0.001998) D10 L0
    error(0.001) D11 D13
    error(0.001) D11 D14
    error(0.001998) D11 D40
    error(0.001998) D12
    error(0.001) D12 D14
    error(0.001998) D12 D41
    error(0.001998) D13 D42
    error(0.001998) D14 D43
    error(0.001998) D29 D32
    error(0.001998) D29 D58
    error(0.001998) D29 L0
    error(0.001998) D30
    error(0.002994) D30 D32
    error(0.002994) D30 D33
    error(0.001998) D30 D59
    error(0.00498004) D31
    error(0.002994) D31 D33
    error(0.001998) D31 D60
    error(0.001998) D32
    error(0.001998) D32 D34
    error(0.002994) D32 D35
    error(0.001998) D32 D61
    error(0.001998) D32 L0
    error(0.001998) D33
    error(0.002994) D33 D35
    error(0.002994) D33 D36
    error(0.001998) D33 D62
    error(0.001998) D34
    error(0.001) D34 D35
    error(0.002994) D34 D37
    error(0.001998) D34 D63
    error(0.00597008) D34 L0
    error(0.001998) D35
    error(0.001998) D35 D37
    error(0.002994) D35 D38
    error(0.001998) D35 D64
    error(0.00794422) D36
    error(0.002994) D36 D38
    error(0.001998) D36 D65
    error(0.001998) D37
    error(0.001) D37 D38
    error(0.002994) D37 D39
    error(0.002994) D37 D40
    error(0.001998) D37 D66
    error(0.001998) D38
    error(0.001998) D38 D40
    error(0.002994) D38 D41
    error(0.001998) D38 D67
    error(0.001998) D39
    error(0.002994) D39 D42
    error(0.001998) D39 D68
    error(0.00597008) D39 L0
    error(0.001998) D40
    error(0.001) D40 D41
    error(0.002994) D40 D42
    error(0.002994) D40 D43
    error(0.001998) D40 D69
    error(0.00794422) D41
    error(0.001998) D41 D43
    error(0.001998) D41 D70
    error(0.001998) D42
    error(0.001998) D42 D71
    error(0.002994) D43
    error(0.001998) D43 D72
    error(0.001) D58 D61
    error(0.001998) D58 D87
    error(0.001) D58 L0
    error(0.001) D59 D61
    error(0.001) D59 D62
    error(0.001998) D59 D88
    error(0.001) D60
    error(0.001) D60 D62
    error(0.001998) D60 D89
    error(0.001) D61 D63
    error(0.001) D61 D64
    error(0.001998) D61 D90
    error(0.001) D62 D64
    error(0.001) D62 D65
    error(0.001998) D62 D91
    error(0.001) D63 D66
    error(0.001998) D63 D92
    error(0.001998) D63 L0
    error(0.001) D64 D66
    error(0.001) D64 D67
    error(0.001998) D64 D93
    error(0.001998) D65
    error(0.001) D65 D67
    error(0.001998) D65 D94
    error(0.001) D66 D68
    error(0.001) D66 D69
    error(0.001998) D66 D95
    error(0.001) D67 D69
    error(0.001) D67 D70
    error(0.001998) D67 D96
    error(0.001) D68 D71
    error(0.001998) D68 D97
    error(0.001998) D68 L0
    error(0.001) D69 D71
    error(0.001) D69 D72
    error(0.001998) D69 D98
    error(0.001998) D70
    error(0.001) D70 D72
    error(0.001998) D70 D99
    error(0.001998) D71 D100
    error(0.001998) D72 D101
    error(0.001998) D87 D90
    error(0.001998) D87 D116
    error(0.001998) D87 L0
    error(0.001998) D88
    error(0.002994) D88 D90
    error(0.002994) D88 D91
    error(0.001998) D88 D117
    error(0.00498004) D89
    error(0.002994) D89 D91
    error(0.001998) D89 D118
    error(0.00398802) D90
    error(0.001998) D90 D92
    error(0.002994) D90 D93
    error(0.001998) D90 D119
    error(0.001998) D91
    error(0.002994) D91 D93
    error(0.002994) D91 D94
    error(0.001998) D91 D120
    error(0.00398802) D92
    error(0.001) D92 D93
    error(0.002994) D92 D95
    error(0.001998) D92 D121
    error(0.00398802) D92 L0
    error(0.001998) D93
    error(0.001998) D93 D95
    error(0.002994) D93 D96
    error(0.001998) D93 D122
    error(0.00794422) D94
    error(0.002994) D94 D96
    error(0.001998) D94 D123
    error(0.001998) D95
    error(0.001) D95 D96
    error(0.002994) D95 D97
    error(0.002994) D95 D98
    error(0.001998) D95 D124
    error(0.001998) D96
    error(0.001998) D96 D98
    error(0.002994) D96 D99
    error(0.001998) D96 D125
    error(0.00398802) D97
    error(0.002994) D97 D100
    error(0.001998) D97 D126
    error(0.00398802) D97 L0
    error(0.001998) D98
    error(0.001) D98 D99
    error(0.002994) D98 D100
    error(0.002994) D98 D101
    error(0.001998) D98 D127
    error(0.00794422) D99
    error(0.001998) D99 D101
    error(0.001998) D99 D128
    error(0.001998) D100
    error(0.001998) D100 D129
    error(0.002994) D101
    error(0.001998) D101 D130
    error(0.001) D116 D119
    error(0.001998) D116 D145
    error(0.001) D116 L0
    error(0.001) D117 D119
    error(0.001) D117 D120
    error(0.001998) D117 D146
    error(0.001) D118
    error(0.001) D118 D120
    error(0.001998) D118 D147
    error(0.001) D119 D121
    error(0.001) D119 D122
    error(0.001998) D119 D148
    error(0.001) D120 D122
    error(0.001) D120 D123
    error(0.001998) D120 D149
    error(0.001) D121 D124
    error(0.001998) D121 D150
    error(0.001998) D121 L0
    error(0.001) D122 D124
    error(0.001) D122 D125
    error(0.001998) D122 D151
    error(0.001998) D123
    error(0.001) D123 D125
    error(0.001998) D123 D152
    error(0.001) D124 D126
    error(0.001) D124 D127
    error(0.001998) D124 D153
    error(0.001) D125 D127
    error(0.001) D125 D128
    error(0.001998) D125 D154
    error(0.001) D126 D129
    error(0.001998) D126 D155
    error(0.001998) D126 L0
    error(0.001) D127 D129
    error(0.001) D127 D130
    error(0.001998) D127 D156
    error(0.001998) D128
    error(0.001) D128 D130
    error(0.001998) D128 D157
    error(0.001998) D129 D158
    error(0.001998) D130 D159
    error(0.002994) D145 D148
    error(0.002994) D145 L0
    error(0.002994) D146 D148
    error(0.002994) D146 D149
    error(0.002994) D147
    error(0.002994) D147 D149
    error(0.002994) D148 D150
    error(0.002994) D148 D151
    error(0.002994) D149 D151
    error(0.002994) D149 D152
    error(0.002994) D150 D153
    error(0.00597008) D150 L0
    error(0.002994) D151 D153
    error(0.002994) D151 D154
    error(0.00597008) D152
    error(0.002994) D152 D154
    error(0.002994) D153 D155
    error(0.002994) D153 D156
    error(0.002994) D154 D156
    error(0.002994) D154 D157
    error(0.002994) D155 D158
    error(0.00597008) D155 L0
    error(0.002994) D156 D158
    error(0.002994) D156 D159
    error(0.00597008) D157
    error(0.002994) D157 D159
    error(0.0118689) L0
    detector D15
    detector D16
    detector D17
    detector D18
    detector D19
    detector D20
    detector D21
    detector D22
    detector D23
    detector D24
    detector D25
    detector D26
    detector D27
    detector D28
    detector D44
    detector D45
    detector D46
    detector D47
    detector D48
    detector D49
    detector D50
    detector D51
    detector D52
    detector D53
    detector D54
    detector D55
    detector D56
    detector D57
    detector D73
    detector D74
    detector D75
    detector D76
    detector D77
    detector D78
    detector D79
    detector D80
    detector D81
    detector D82
    detector D83
    detector D84
    detector D85
    detector D86
    detector D102
    detector D103
    detector D104
    detector D105
    detector D106
    detector D107
    detector D108
    detector D109
    detector D110
    detector D111
    detector D112
    detector D113
    detector D114
    detector D115
    detector D131
    detector D132
    detector D133
    detector D134
    detector D135
    detector D136
    detector D137
    detector D138
    detector D139
    detector D140
    detector D141
    detector D142
    detector D143
    detector D144
    """
    )

    split_mwpm = SplitMatching(dem, dem_z=dem_z, dem_x=dem_x)

    syndrome = np.array(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
    )

    prediction = split_mwpm.decode(syndrome)

    assert prediction.shape == (1,)
    assert prediction.sum() == 0

    return
