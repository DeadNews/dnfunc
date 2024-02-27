.PHONY: all clean test

vendor-up:
	@echo "Update vendor packages that are distributed as a single source file."

	wget -O vendor/vendor/CSMOD.py https://raw.githubusercontent.com/fdar0536/VapourSynth-Contra-Sharpen-mod/master/CSMOD.py
	wget -O vendor/vendor/descale.py https://raw.githubusercontent.com/Irrational-Encoding-Wizardry/descale/master/descale.py
	wget -O vendor/vendor/finesharp.py https://gist.githubusercontent.com/4re/8676fd350d4b5b223ab9/raw/892023c42882eee7f4ec3b18a9b1cff43fdd9f40/finesharp.py
	wget -O vendor/vendor/fvsfunc.py https://raw.githubusercontent.com/Irrational-Encoding-Wizardry/fvsfunc/master/fvsfunc.py
	wget -O vendor/vendor/kagefunc.py https://raw.githubusercontent.com/Irrational-Encoding-Wizardry/kagefunc/master/kagefunc.py
	wget -O vendor/vendor/muvsfunc.py https://raw.githubusercontent.com/WolframRhodium/muvsfunc/master/muvsfunc.py
