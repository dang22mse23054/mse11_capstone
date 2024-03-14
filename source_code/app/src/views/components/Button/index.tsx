import React, { CSSProperties, forwardRef } from 'react';
import { makeStyles, useTheme } from '@material-ui/core/styles';
import { Button, IconButton, ButtonProps } from '@material-ui/core';
import { green, lime, grey, lightGreen, orange, purple, blue, lightBlue, teal, yellow, blueGrey, brown, indigo, red, pink, cyan } from '@material-ui/core/colors';

export const ColorNames = [
	'green', 'grey', 'lightGreen', 'orange', 'purple', 'blue', 'lightBlue',
	'teal', 'yellow', 'blueGrey', 'brown', 'indigo', 'red', 'pink',
	'lime', 'primary', 'secondary', 'default', 'cyan'
] as const;

const ColorKeys = {
	GREEN: ColorNames[0],
	GREY: ColorNames[1],
	LIGHTGREEN: ColorNames[2],
	ORANGE: ColorNames[3],
	PURPLE: ColorNames[4],
	BLUE: ColorNames[5],
	LIGHTBLUE: ColorNames[6],
	TEAL: ColorNames[7],
	YELLOW: ColorNames[8],
	BLUEGREY: ColorNames[9],
	BROWN: ColorNames[10],
	INDIGO: ColorNames[11],
	RED: ColorNames[12],
	PINK: ColorNames[13],
	LIME: ColorNames[14],
	PRIMARY: ColorNames[15],
	SECONDARY: ColorNames[16],
	DEFAULT: ColorNames[17],
	CYAN: ColorNames[18],
};

const ThemeKeys = [
	ColorKeys.DEFAULT,
	ColorKeys.PRIMARY,
	ColorKeys.SECONDARY,
];

export const ColorLevels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 'A100', 'A200', 'A400', 'A700'] as const;

const COLOR = {
	[ColorKeys.GREEN]: green,
	[ColorKeys.GREY]: grey,
	[ColorKeys.LIGHTGREEN]: lightGreen,
	[ColorKeys.ORANGE]: orange,
	[ColorKeys.PURPLE]: purple,
	[ColorKeys.BLUE]: blue,
	[ColorKeys.LIGHTBLUE]: lightBlue,
	[ColorKeys.TEAL]: teal,
	[ColorKeys.YELLOW]: yellow,
	[ColorKeys.BLUEGREY]: blueGrey,
	[ColorKeys.BROWN]: brown,
	[ColorKeys.INDIGO]: indigo,
	[ColorKeys.RED]: red,
	[ColorKeys.PINK]: pink,
	[ColorKeys.LIME]: lime,
	[ColorKeys.CYAN]: cyan,
};

export interface IProps extends ButtonProps {
	btnType?: 'default' | 'icon'
	btnColor?: typeof ColorNames[number];
	btnContrast?: typeof ColorLevels[number];
	disabled?: boolean
	className?: any
	variant?: 'outlined' | 'contained'
	style?: CSSProperties
}

const defaultProps: IProps = {
	btnType: 'default',
	btnContrast: 700
};

const ColorButton: React.FC<IProps> = forwardRef((props, ref) => {
	const theme = useTheme();
	const {
		btnType,
		btnColor = (btnType == 'icon' ? ColorKeys.DEFAULT : ColorKeys.PRIMARY),
		btnContrast, className, ...restProps
	} = props;
	const { variant, style, disabled } = restProps;

	const isUseTheme = ThemeKeys.includes(btnColor);

	const constrastId = ColorLevels.indexOf(btnContrast);

	let constrastHoverId: number;

	const ButtonType = btnType == 'default' ? Button : IconButton;

	// ---- Config Style ---- //
	if (variant !== 'outlined') {
		constrastHoverId = constrastId + 2;
		constrastHoverId = constrastHoverId > (ColorLevels.length - 1) ? 9 : constrastHoverId;
	}

	let customStyle = {};

	if (!isUseTheme) {
		customStyle = {
			root: {
				color: COLOR[btnColor][ColorLevels[constrastId]] ? COLOR[btnColor][ColorLevels[constrastId]] : (COLOR[btnColor]['A400'] ? COLOR[btnColor]['A400'] : COLOR[btnColor][700]),
				'&:hover': {
					backgroundColor: COLOR[btnColor][900] + '0a',
				}
			},
		};

		if (props.btnType == 'default') {
			customStyle = {
				...customStyle,

				outlined: {
					color: COLOR[btnColor][ColorLevels[constrastId]] ? COLOR[btnColor][ColorLevels[constrastId]] : (COLOR[btnColor]['A400'] ? COLOR[btnColor]['A400'] : COLOR[btnColor][600]),
					backgroundColor: 'none',
					borderColor: (COLOR[btnColor][ColorLevels[constrastId]] ? COLOR[btnColor][ColorLevels[constrastId]] : (COLOR[btnColor]['A400'] ? COLOR[btnColor]['A400'] : COLOR[btnColor][600])) + '80',
					'&:hover': {
						backgroundColor: COLOR[btnColor][900] + '0a',
						borderColor: COLOR[btnColor][500],
					}
				},
				contained: {
					color: theme.palette.getContrastText(COLOR[btnColor][ColorLevels[constrastId]]),
					backgroundColor: COLOR[btnColor][ColorLevels[constrastId]],
					'&:hover': {
						backgroundColor: COLOR[btnColor][ColorLevels[constrastHoverId]],
					}
				},
			};
		}
	}
	// ---- Finish Config Style ---- //

	const classes = makeStyles(customStyle)(props);
	const classNames = `${className || ''} ${isUseTheme ? '' : `${classes.root} ${variant === 'outlined' ? classes.outlined : variant === 'contained' ? classes.contained : ''}`}`;

	return (
		<ButtonType ref={ref} {...(style ? { style } : {})} {...(disabled ? { disabled } : {})} {...restProps}
			className={`${classNames}`} color={isUseTheme ? btnColor : 'primary'} >{props.children}</ButtonType>
	);
});

ColorButton.defaultProps = defaultProps;

export default ColorButton;