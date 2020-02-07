from greek_data_prep.clean_data import clean_tokens, clean_texts, CHARS_TO_REPLACE, CHARS_TO_REMOVE, parse_xml

def test_clean_tokens():
    # it should not remove tokens consisting only of greek chars
    tokens = "ἤγαγε Σιδονίηθεν, ἐπιπλὼς εὐρέα πόντον".split(' ')
    assert clean_tokens(tokens, CHARS_TO_REMOVE, CHARS_TO_REPLACE) == tokens
    # it should strip out \t\r\n
    assert clean_tokens('\t\t\t\t\tγνώμαις ὑμεῖς'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['γνώμαις', 'ὑμεῖς']
    # it should remove unwanted chars
    assert clean_tokens([CHARS_TO_REMOVE], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == []
    assert clean_tokens('Αἴης. » οὐ γὰρ τὸ τοῦ Ἀξιοῦ ὕδωρ κάλλιστον ★ τῆς γῆς τῇ ὄψει κίδναται, ἀλλὰ τῆς γῆς τῷ'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == 'Αἴης. " οὐ γὰρ τὸ τοῦ Ἀξιοῦ ὕδωρ κάλλιστον  τῆς γῆς τῇ ὄψει κίδναται, ἀλλὰ τῆς γῆς τῷ'.split(' ')
    ## it should replace chars
    ##TODO find out why † is not present in the output
    ## assert format_texts([['∠ △ ῑ ✝ \ufeff ']], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['Δ Δ ϊ † ']
    assert clean_tokens('∠ △ ῑ \ufeff '.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['Δ', 'Δ', 'ϊ', ' ']
    assert clean_tokens('τριγώνου τὸ U+2220´ ἐστιν τοῦ ὑπὸ ΑΓ, Β∠, τουτέστι ποδῶν'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == 'τριγώνου τὸ ἐστιν τοῦ ὑπὸ ΑΓ, ΒΔ, τουτέστι ποδῶν'.split(' ')
    # it should tokens with latin chars and 
    assert clean_tokens('Soph. Ant. 104 ad 109'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == []
    # it should remove tokens with digits
    assert clean_tokens('4.227-230 4.351-352'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == []
    # it should remove inter-word hyphens and en-dashes
    assert clean_tokens('- — ἐκακούργη- καθά-περ ἱκανῶς—οὐδέ'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['-', '—', 'ἐκακούργη', 'καθάπερ', 'ἱκανῶς—οὐδέ']
    # it should remove other forms of whitespace
    assert clean_tokens('\tδύναμιν\tἔμαθον, \t \n\r'.split(' '), CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['δύναμιν ἔμαθον,', '', '']

def test_format_texts():
    # it should remove empty lines
    assert clean_texts(['\n\n\n\n\n\n\n\n\n\n\n\nὍσα μὲν οὖν Ῥωμαίων τῷ γένει ἐν τοῖς πολέμοις'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['Ὅσα μὲν οὖν Ῥωμαίων τῷ γένει ἐν τοῖς πολέμοις']
    # it should curly braces and their contents 
    assert clean_texts(['ἐξανέψιοι καὶ {ἐξ}ἀνεψιοὶ διαφέρει. καὶ ⌞ὅτι⌟ βαρυτόνως \n⌞Ἀττικοὶ⌟ προφέρονται τοὔνομα φησὶ {γὰρ} Τρύφων'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['ἐξανέψιοι καὶ ἀνεψιοὶ διαφέρει. καὶ ὅτι βαρυτόνως Ἀττικοὶ προφέρονται τοὔνομα φησὶ Τρύφων']
    # ... but not if their content is over 40 chars
    assert clean_texts(['πρὸς τὸν βίον. {βιοῦν μὲν φήσομεν ἀνθρώπους, ζῆν δὲ καὶ τὸ\nἄλογον ζῷον}'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['πρὸς τὸν βίον. βιοῦν μὲν φήσομεν ἀνθρώπους, ζῆν δὲ καὶ τὸ ἄλογον ζῷον']
    # however, it should retain square brackets and their contents
    assert clean_texts(['\t\t\t\t\tἀνεγνωσμένον παρʼ ὑμῖν ἐψηφίσαντο ψήφισμα, [μὴ]\nμὴ \ndel. Sauppe: ret. Blass,'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['ἀνεγνωσμένον παρʼ ὑμῖν ἐψηφίσαντο ψήφισμα, [μὴ] μὴ']
    text_with_square_brackets = 'ἐὰν γ(ὰρ) ἔχηι ὑπὸ νόσο[υ τὸ] ζῶιον ηελ[. . . .]ν'
    assert clean_texts([text_with_square_brackets], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == [text_with_square_brackets]
    # it should remove empty parentheses
    # TODO this tests demonstrate the undesired position of the comma after the parens are removed
    assert clean_texts(['‟ὁ οὐρανός μοι θρόνος (  ), ἡ δὲ γῆ '], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['"ὁ οὐρανός μοι θρόνος , ἡ δὲ γῆ']
    # it should remove text between ⟦ and ⟧ and the braces themselves
    assert clean_texts(['⟦Πάντα δὲ τὰ παρηγγελμένα τοῖς περὶ τὸν Πέτρον\n\nσυντόμως ἐξήγγειλαν. Μετὰ δὲ ταῦτα καὶ αὐτὸς ὁ Ἰησοῦς \n\nἀπὸ ἀνατολῆς καὶ ἀχρὶ δύσεως ἐξαπέστειλεν διʼ αὐτῶν \n\nτὸ ἱερὸν καὶ ἄφθαρτον κήρυγμα τῆς αἰωνίου        σωτηρίας.⟧'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['']
    # it should remove empty left and right arrows
    assert clean_texts(['καὶ διὰ τούτων <> κήπων, τρέφοντες'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['καὶ διὰ τούτων κήπων, τρέφοντες']
    # it should remove repeated whitespace
    assert clean_texts(['          εἰθισμένας μεταπίπτειν. \n\n      \n\n      \n\n        '], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['εἰθισμένας μεταπίπτειν.']
    # it should concatenate texts to a single string
    assert clean_texts(['Α\nΑΑ\nΑΒ', 'Β\nΒΑ\nΒΒ', 'Γ\nΓΑ\nΓΒ'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['Α ΑΑ ΑΒ', 'Β ΒΑ ΒΒ', 'Γ ΓΑ ΓΒ']
    # it should remove beta code
    assert clean_texts(['kai\\ w(s ta\\s fwna\\s au)tw=n mete/bale kai\\ o( to/pos, e )n w(=| tou=to\nge/gone, *babulw\\n e)klh/qh.'], CHARS_TO_REMOVE, CHARS_TO_REPLACE) == ['']

    # longer 'integration' tests
    # TODO this test illusrates how greek text for the apparatus criticus can find its way into the text...
    text_0 = ['\n    \n\n      \n\n        \n\n\n\n\nεἰ μὴ συνῄδη, ὦ βουλή, τοῖς κατηγόροις βουλομένοις ἐκ παντὸς τρόπου κακῶς ἐμὲ ποιεῖν, πολλὴν ἂν αὐτοῖς χάριν        εἶχον ταύτης τῆς κατηγορίας· ἡγοῦμαι γὰρ τοῖς ἀδίκως διαβεβλημένοις τούτους εἶναι μεγίστων ἀγαθῶν αἰτίους, οἵτινες ἂν αὐτοὺς ἀναγκάζωσιν εἰς ἔλεγχον τῶν αὐτοῖς βεβιωμένων      καταστῆναι.\n \n\n        \nἐγὼ γὰρ οὕτω σφόδρα ἐμαυτῷ πιστεύω, ὥστʼ ἐλπίζω καὶ εἴ τις πρός με τυγχάνει ἀηδῶς \nἢ κακῶς\nἢ κακῶς\n del. Reiske\n                διακείμενος, ἐπειδὰν ἐμοῦ λέγοντος ἀκούσῃ περὶ τῶν πεπραγμένων, μεταμελήσειν αὐτῷ καὶ πολὺ βελτίω με εἰς τὸν λοιπὸν χρόνον ἡγήσεσθαι. \n\n\n        \nἀξιῶ δέ, ὦ        βουλή, ἐὰν μὲν τοῦτο μόνον ὑμῖν ἐπιδείξω, ὡς εὔνους εἰμὶ τοῖς καθεστηκόσι πράγμασι καὶ ὡς ἠνάγκασμαι τῶν αὐτῶν κινδύνων μετέχειν ὑμῖν, μηδέν πώ μοι πλέον εἶναι· ἐὰν δὲ         φαίνωμαι \nκαὶ\nκαὶ\n add. Reiske\n περὶ τὰ ἄλλα μετρίως βεβιωκὼς καὶ πολὺ παρὰ τὴν δόξαν καὶ παρὰ τοὺς λόγους τοὺς τῶν ἐχθρῶν, δέομαι ὑμῶν ἐμὲ μὲν δοκιμάζειν,         τούτους δὲ ἡγεῖσθαι χείρους εἶναι. πρῶτον δὲ ἀποδείξω ὡς οὐχ ἵππευον οὐδʼ ἐπεδήμουν ἐπὶ τῶν τριάκοντα, οὐδὲ μετέσχον τῆς τότε πολιτείας. \n\n\n\n\n  \nἡμᾶς         γὰρ ὁ πατὴρ πρὸ τῆς ἐν Ἑλλησπόντῳ συμφορᾶς ὡς Σάτυρον τὸν ἐν τῷ Πόντῳ διαιτησομένους ἐξέπεμψε, καὶ οὔτε τῶν τειχῶν καθαιρουμένων \nἐπεδημοῦμεν\nὅτι\n Kayser: \n          διότι\n MSS.\n οὔτε μεθισταμένης τῆς πολιτείας, ἀλλʼ ἤλθομεν πρὶν τοὺς ἀπὸ Φυλῆς εἰς τὸν Πειραιᾶ κατελθεῖν πρότερον πένθʼ ἡμέραις. ']
    output_0 = ['εἰ μὴ συνῄδη, ὦ βουλή, τοῖς κατηγόροις βουλομένοις ἐκ παντὸς τρόπου κακῶς ἐμὲ ποιεῖν, πολλὴν ἂν αὐτοῖς χάριν εἶχον ταύτης τῆς κατηγορίας· ἡγοῦμαι γὰρ τοῖς ἀδίκως διαβεβλημένοις τούτους εἶναι μεγίστων ἀγαθῶν αἰτίους, οἵτινες ἂν αὐτοὺς ἀναγκάζωσιν εἰς ἔλεγχον τῶν αὐτοῖς βεβιωμένων καταστῆναι. ἐγὼ γὰρ οὕτω σφόδρα ἐμαυτῷ πιστεύω, ὥστʼ ἐλπίζω καὶ εἴ τις πρός με τυγχάνει ἀηδῶς ἢ κακῶς ἢ κακῶς διακείμενος, ἐπειδὰν ἐμοῦ λέγοντος ἀκούσῃ περὶ τῶν πεπραγμένων, μεταμελήσειν αὐτῷ καὶ πολὺ βελτίω με εἰς τὸν λοιπὸν χρόνον ἡγήσεσθαι. ἀξιῶ δέ, ὦ βουλή, ἐὰν μὲν τοῦτο μόνον ὑμῖν ἐπιδείξω, ὡς εὔνους εἰμὶ τοῖς καθεστηκόσι πράγμασι καὶ ὡς ἠνάγκασμαι τῶν αὐτῶν κινδύνων μετέχειν ὑμῖν, μηδέν πώ μοι πλέον εἶναι· ἐὰν δὲ φαίνωμαι καὶ καὶ περὶ τὰ ἄλλα μετρίως βεβιωκὼς καὶ πολὺ παρὰ τὴν δόξαν καὶ παρὰ τοὺς λόγους τοὺς τῶν ἐχθρῶν, δέομαι ὑμῶν ἐμὲ μὲν δοκιμάζειν, τούτους δὲ ἡγεῖσθαι χείρους εἶναι. πρῶτον δὲ ἀποδείξω ὡς οὐχ ἵππευον οὐδʼ ἐπεδήμουν ἐπὶ τῶν τριάκοντα, οὐδὲ μετέσχον τῆς τότε πολιτείας. ἡμᾶς γὰρ ὁ πατὴρ πρὸ τῆς ἐν Ἑλλησπόντῳ συμφορᾶς ὡς Σάτυρον τὸν ἐν τῷ Πόντῳ διαιτησομένους ἐξέπεμψε, καὶ οὔτε τῶν τειχῶν καθαιρουμένων ἐπεδημοῦμεν ὅτι διότι οὔτε μεθισταμένης τῆς πολιτείας, ἀλλʼ ἤλθομεν πρὶν τοὺς ἀπὸ Φυλῆς εἰς τὸν Πειραιᾶ κατελθεῖν πρότερον πένθʼ ἡμέραις.']
    assert clean_texts(text_0, CHARS_TO_REMOVE, CHARS_TO_REPLACE) == output_0
    # TODO the inclusion of note tags also results in greek text from the apparatus criticus entering the main text here...this has since been fixed by better xml parsing
    text_1 = ['\n\n\n\n\n\n\n\n\n\n\n\nΣωρκανὸν \n\n\nΣωρκανὸν\n] \nεἰς\n                        ὁρκάνην\n Madvigius (coll. Alciphr. Epist. 1,         18: \nἰχθύων οὓς ἐγκολπίζεται τῇ σαγήνῃ\n).\n                        \nΝωρβανὸν\n Emperius. Malim \nΣωρανὸν\n, sed nihil certum\n ἐγκολπίσασθαι\n                                 καὶ φιλίαν τιμᾶν\n\n\n\n καὶ \n\n\nτιμᾶν καὶ\n] margo cod. E\n                     exhibet \nτοι\n Fort. del. \nκαὶ\n              praegressum et corrig. \nφιλίαν μέντοι\n                        μετιέναι κἑ\n\n\n μετιέναι καὶ προσδέχεσθαι καὶ γεωργεῖν,\n                  πολλοῖς μὲν ἰδίᾳ       πολλοῖς δὲ καὶ δημοσίᾳ χρήσιμον καὶ ἔγκαρπον\n                  γενησομένην, φιλοκάλων ἐστὶ καὶ πολιτικῶν καὶ φιλανθρώπων οὐχ ὡς ἔνιοι\n                  νομίζουσι         φιλοδόξων·ʼ ἀλλὰ καὶ τοὐναντίον, φιλόδοξός ἐστι\n\n καὶ ψοφοδεὴς; ὁ φεύγων καὶ φοβούμενος ἀκοῦσαι\n                  λιπαρὴς \n\n\nλιπαρὴς\n] \nλιπαρὴς         θεραπευτὴς\n R.\n                     Malim \nλιπαρὴς θαυμαστὴς\n aut \nθιασώτης\n\n\n τῶν ἐν ἐξουσίᾳ καὶ\n                  θεραπευτικός \n\n\n          θεραπευτικὸς\n] ex praeced. natum. \nπολιτικὸς\n R. Malim \nπρακτικὸς\n cf. Praefat. p. XCIII et prag. 777a\n. ἐπεὶ τί\n                  φησιν ἀνὴρ             θεραπευτικὸς καὶ φιλοσοφίας δεόμενος; Σίμων οὖν \n\n\nοὖν\n *: \nεἰ\n cf. p. 831b:']
    output_1 = ['Σωρκανὸν Σωρκανὸν ] εἰς ὁρκάνην ἰχθύων οὓς ἐγκολπίζεται τῇ σαγήνῃ ). Νωρβανὸν Σωρανὸν , ἐγκολπίσασθαι καὶ φιλίαν τιμᾶν καὶ τιμᾶν καὶ ] τοι καὶ φιλίαν μέντοι μετιέναι κἑ μετιέναι καὶ προσδέχεσθαι καὶ γεωργεῖν, πολλοῖς μὲν ἰδίᾳ πολλοῖς δὲ καὶ δημοσίᾳ χρήσιμον καὶ ἔγκαρπον γενησομένην, φιλοκάλων ἐστὶ καὶ πολιτικῶν καὶ φιλανθρώπων οὐχ ὡς ἔνιοι νομίζουσι φιλοδόξων·ʼ ἀλλὰ καὶ τοὐναντίον, φιλόδοξός ἐστι καὶ ψοφοδεὴς; ὁ φεύγων καὶ φοβούμενος ἀκοῦσαι λιπαρὴς λιπαρὴς ] λιπαρὴς θεραπευτὴς λιπαρὴς θαυμαστὴς θιασώτης τῶν ἐν ἐξουσίᾳ καὶ θεραπευτικός θεραπευτικὸς ] πολιτικὸς πρακτικὸς . ἐπεὶ τί φησιν ἀνὴρ θεραπευτικὸς καὶ φιλοσοφίας δεόμενος; Σίμων οὖν οὖν : εἰ']
    assert clean_texts(text_1, CHARS_TO_REMOVE, CHARS_TO_REPLACE) == output_1
    text_2 = ['\n\n \n\n \n   \nἈρτοξέρξης\n \n   \n \n\n \n\n \n ὁ μὲν πρῶτος Ἀρτοξέρξης, τῶν ἐν Πέρσαις βασιλέων πραότητι καὶ μεγαλοψυχίᾳ πρωτεύσας,           Μακρόχειρ ἐπεκαλεῖτο τὴν δεξιὰν μείζονα τῆς ἑτέρας ἔχων, Ξέρξου δὲ ἦν υἱός· ὁ δὲ δεύτερος, περὶ οὗ τάδε γράφεται, Μνήμων ἐπικληθείς ἐκ θυγατρὸς ἦν ἐκεῖνον. Δαρείου γὰρ         καὶ Παρυσάτιδος \n παῖδες ἐγένοντο τέσσαρες, πρεσβύτατος μὲν Ἀρτοξέρξης, μετʼ ἐκεῖνον δὲ Κῦρος, νεώτεροι δὲ τούτων Ὀστάνης καὶ Ὀξάθρης.\n \n\n \n ὁ μὲν οὖν           Κῦρος ἀπὸ Κύρου τοῦ παλαιοῦ τοὔνομα ἔσχεν, ἐκείνῳ δὲ ἀπὸ τοῦ ἡλίου γενέσθαι φασί Κῦρον γὰρ καλεῖν Πέρσας τὸν ἥλιον. ὁ δὲ Ἀρτοξέρξης Ἀρσίκας πρότερον ἐκαλεῖτο·']
    output_2 = ['Ἀρτοξέρξης ὁ μὲν πρῶτος Ἀρτοξέρξης, τῶν ἐν Πέρσαις βασιλέων πραότητι καὶ μεγαλοψυχίᾳ πρωτεύσας, Μακρόχειρ ἐπεκαλεῖτο τὴν δεξιὰν μείζονα τῆς ἑτέρας ἔχων, Ξέρξου δὲ ἦν υἱός· ὁ δὲ δεύτερος, περὶ οὗ τάδε γράφεται, Μνήμων ἐπικληθείς ἐκ θυγατρὸς ἦν ἐκεῖνον. Δαρείου γὰρ καὶ Παρυσάτιδος παῖδες ἐγένοντο τέσσαρες, πρεσβύτατος μὲν Ἀρτοξέρξης, μετʼ ἐκεῖνον δὲ Κῦρος, νεώτεροι δὲ τούτων Ὀστάνης καὶ Ὀξάθρης. ὁ μὲν οὖν Κῦρος ἀπὸ Κύρου τοῦ παλαιοῦ τοὔνομα ἔσχεν, ἐκείνῳ δὲ ἀπὸ τοῦ ἡλίου γενέσθαι φασί Κῦρον γὰρ καλεῖν Πέρσας τὸν ἥλιον. ὁ δὲ Ἀρτοξέρξης Ἀρσίκας πρότερον ἐκαλεῖτο·']
    assert clean_texts(text_2, CHARS_TO_REMOVE, CHARS_TO_REPLACE) == output_2

def test_parse_xml():
    xml_0 = '''<text xml:lang="grc">
<body>
    <div type="edition" xml:lang="grc" n="urn:cts:greekLit:tlg0007.tlg088.perseus-grc3">
<pb xml:id="v.4.p.492"/>
        <div type="textpart" subtype="section" n="1">
            <p rend="indent">
                <milestone unit="stephpage" xml:id="stephpage-345c" n="345c"/>
<gap reason="lost"/>
                <note resp="ed" place="unspecified" anchored="true" xml:lang="eng">Xylander was the first to suggest a lacuna at the beginning.</note> Ταῦτʼ ὀρθῶς μὲν ἐκεῖνος εἶπε πρὸς τοὺς μεθʼ ἑαυτὸν στρατηγούς, οἷς πάροδον ἐπὶ τὰς
ὕστερον πράξεις ἔδωκεν ἐξελάσας τὸν βάρβαρον
καὶ τὴν Ἑλλάδʼ ἐλευθερώσας· ὀρθῶς δʼ εἰρήσεται
καὶ πρὸς τοὺς ἐπὶ τοῖς λόγοις μέγα φρονοῦντας·
ἂν γάρ ἀνέλῃς τοὺς πράττοντας, οὐχ ἕξεις τοὺς
γράφοντας. ἄνελε τὴν Περικλέους πολιτείαν καὶ
<milestone unit="stephpage" xml:id="stephpage-345d" n="345d"/> τὰ ναύμαχα πρὸς Ῥίῳ Φορμίωνος τρόπαια καὶ τὰς
περὶ Κύθηρα καὶ Μέγαρα καὶ Κόρινθον ἀνδραγαθίας
Νικίου καὶ τὴν Δημοσθένους Πύλον καὶ τοὺς Κλέωνος τετρακοσίους αἰχμαλώτους καὶ Τολμίδαν <note resp="ed" place="unspecified" anchored="true" xml:lang="eng"><foreign xml:lang="grc">Τολμίδαν</foreign> Xylander: <foreign xml:lang="grc">Τολμίαν</foreign>.</note> Πελοπόννησον περιπλέοντα καὶ Μυρωνίδην νικῶντα Βοιωτοὺς ἐν Οἰνοφύτοις, καὶ Θουκυδίδης σοι διαγέγραπται. ἄνελε τὰ περὶ Ἑλλήσποντον
                Ἀλκιβιάδου νεανιεύματα καὶ τὰ πρὸς Λέσβῳ <note resp="ed" place="unspecified" anchored="true" xml:lang="eng"><foreign xml:lang="grc">Λέσβῳ</foreign> F.C.B.: <foreign xml:lang="grc">Λέσβον</foreign>.</note> Θρασύλλου
                καὶ τὴν ὑπὸ Θηραμένους τῆς ὀλιγαρχίας κατάλυσιν καὶ Θρασύβουλον καὶ Ἀρχῖνον<note resp="ed" place="unspecified" anchored="true" xml:lang="eng"><foreign xml:lang="grc">Ἀρχῖνον</foreign> Taylor, from 835 F: <foreign xml:lang="grc">ἄρχιππον</foreign>.</note> καὶ τοὺς ἀπὸ
Φυλῆς ἑβδομήκοντα κατὰ τῆς Σπαρτιατῶν
ἡγεμονίας ἀνισταμένους καὶ Κόνωνα πάλιν ἐμβιβάζοντα
<pb xml:id="v.4.p.494"/>
<milestone unit="stephpage" xml:id="stephpage-345e" n="345e"/> τὰς Ἀθήνας εἰς τὴν θάλατταν, καὶ Κράτιππος
ἀνῄρηται. 
</p>
</div>
</body>
</text>'''
    output_0 = '\n\n\n\n\n\n\n\n Ταῦτʼ ὀρθῶς μὲν ἐκεῖνος εἶπε πρὸς τοὺς μεθʼ ἑαυτὸν στρατηγούς, οἷς πάροδον ἐπὶ τὰς\nὕστερον πράξεις ἔδωκεν ἐξελάσας τὸν βάρβαρον\nκαὶ τὴν Ἑλλάδʼ ἐλευθερώσας· ὀρθῶς δʼ εἰρήσεται\nκαὶ πρὸς τοὺς ἐπὶ τοῖς λόγοις μέγα φρονοῦντας·\nἂν γάρ ἀνέλῃς τοὺς πράττοντας, οὐχ ἕξεις τοὺς\nγράφοντας. ἄνελε τὴν Περικλέους πολιτείαν καὶ\n τὰ ναύμαχα πρὸς Ῥίῳ Φορμίωνος τρόπαια καὶ τὰς\nπερὶ Κύθηρα καὶ Μέγαρα καὶ Κόρινθον ἀνδραγαθίας\nΝικίου καὶ τὴν Δημοσθένους Πύλον καὶ τοὺς Κλέωνος τετρακοσίους αἰχμαλώτους καὶ Τολμίδαν  Πελοπόννησον περιπλέοντα καὶ Μυρωνίδην νικῶντα Βοιωτοὺς ἐν Οἰνοφύτοις, καὶ Θουκυδίδης σοι διαγέγραπται. ἄνελε τὰ περὶ Ἑλλήσποντον\n                Ἀλκιβιάδου νεανιεύματα καὶ τὰ πρὸς Λέσβῳ  Θρασύλλου\n                καὶ τὴν ὑπὸ Θηραμένους τῆς ὀλιγαρχίας κατάλυσιν καὶ Θρασύβουλον καὶ Ἀρχῖνον καὶ τοὺς ἀπὸ\nΦυλῆς ἑβδομήκοντα κατὰ τῆς Σπαρτιατῶν\nἡγεμονίας ἀνισταμένους καὶ Κόνωνα πάλιν ἐμβιβάζοντα\n\n τὰς Ἀθήνας εἰς τὴν θάλατταν, καὶ Κράτιππος\nἀνῄρηται. \n\n\n\n'
    assert parse_xml(xml_0) == output_0
    
    xml_1 = '''<text>
<body>
<div1 type="chapter">
<pb id="v.5.p.1"/>
<milestone unit="stephpage" id="stephpage-776a" n="776a"/>
<div2 type="section" n="1">
<p>Σωρκανὸν <note resp="ed" place="unspecified" anchored="yes">
<foreign lang="greek">Σωρκανὸν</foreign>] <foreign lang="greek">εἰς
                        ὁρκάνην</foreign> Madvigius (coll. Alciphr. Epist. 1, 18: <foreign lang="greek">ἰχθύων οὓς ἐγκολπίζεται τῇ σαγήνῃ</foreign>).
                        <foreign lang="greek">Νωρβανὸν</foreign> Emperius. Malim <foreign lang="greek">Σωρανὸν</foreign>, sed nihil certum</note> ἐγκολπίσασθαι
                  καὶ φιλίαν τιμᾶν
<milestone unit="stephpage" id="stephpage-776b" n="776b"/>
<milestone unit="tlnum" n="5"/> καὶ <note resp="ed" place="unspecified" anchored="yes">
<foreign lang="greek">τιμᾶν καὶ</foreign>] margo cod. E
                     exhibet <foreign lang="greek">τοι</foreign> Fort. del. <foreign lang="greek">καὶ</foreign> praegressum et corrig. <foreign lang="greek">φιλίαν μέντοι
                        μετιέναι κἑ</foreign>
</note> μετιέναι καὶ προσδέχεσθαι καὶ γεωργεῖν,
                  πολλοῖς μὲν ἰδίᾳ πολλοῖς δὲ καὶ δημοσίᾳ χρήσιμον καὶ ἔγκαρπον
                  γενησομένην, φιλοκάλων ἐστὶ καὶ πολιτικῶν καὶ φιλανθρώπων οὐχ ὡς ἔνιοι
                  νομίζουσι φιλοδόξων·ʼ ἀλλὰ καὶ τοὐναντίον, φιλόδοξός ἐστι
<milestone unit="tlnum" n="10"/> καὶ ψοφοδεὴς; ὁ φεύγων καὶ φοβούμενος ἀκοῦσαι
                  λιπαρὴς <note resp="ed" place="unspecified" anchored="yes">
<foreign lang="greek">λιπαρὴς</foreign>] <foreign lang="greek">λιπαρὴς θεραπευτὴς</foreign> R.
                     Malim <foreign lang="greek">λιπαρὴς θαυμαστὴς</foreign> aut <foreign lang="greek">θιασώτης</foreign>
</note>
</div2>
</div>
</body>
</text>'''
    output_1 = '\n\n\n\n\n\nΣωρκανὸν  ἐγκολπίσασθαι\n                  καὶ φιλίαν τιμᾶν\n\n καὶ  μετιέναι καὶ προσδέχεσθαι καὶ γεωργεῖν,\n                  πολλοῖς μὲν ἰδίᾳ πολλοῖς δὲ καὶ δημοσίᾳ χρήσιμον καὶ ἔγκαρπον\n                  γενησομένην, φιλοκάλων ἐστὶ καὶ πολιτικῶν καὶ φιλανθρώπων οὐχ ὡς ἔνιοι\n                  νομίζουσι φιλοδόξων·ʼ ἀλλὰ καὶ τοὐναντίον, φιλόδοξός ἐστι\n καὶ ψοφοδεὴς; ὁ φεύγων καὶ φοβούμενος ἀκοῦσαι\n                  λιπαρὴς \n\n\n\n'
    assert parse_xml(xml_1) == output_1
