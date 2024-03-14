import React, { CSSProperties, FC } from 'react';
import NotchedOutline from '@material-ui/core/OutlinedInput/NotchedOutline';
import { createStyles, Theme, makeStyles } from '@material-ui/core/styles';
import { Box, FormControl, InputLabel, Typography } from '@material-ui/core';
import { red, grey } from '@material-ui/core/colors';
import clsx from 'clsx';

interface IProps {
	label?: string
	placeholder?: any
	style?: CSSProperties
	contentStyle?: CSSProperties
	children: any
	isError?: boolean
	isShrink?: any
	size?: 'small' | 'medium'
	variant?: 'standard' | 'filled' | 'outlined'
	disabled?: boolean
}

export const ControlOutline: FC<IProps> = ({ variant, isError, ...props }) => {
	const [labelWidth, setLabelWidth] = React.useState(0);
	const labelRef = React.useRef(null);

	React.useEffect(() => {
		const labelNode = labelRef.current;
		const paddingInline = 10;
		setLabelWidth(labelNode != null ? labelNode.offsetWidth + paddingInline : 0);
	}, [label]);

	variant = variant || 'outlined';
	// const marginTop = variant == 'standard' ? '15px' : '0';
	const marginTop = 0;
	const padding = variant == 'standard' ? '3.5px 0' : '18.5px 14px';
	const randomStr = Math.random().toString(36).substr(2, 5);
	const customNotchedOutlineClassName = `customNotchedOutline-${randomStr}`;
	const customContentClassName = `customContent-${randomStr}`;

	const useStyles = makeStyles((theme: Theme) => createStyles({
		placeholder: {
			color: grey[500],
			// fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif;',

			// ['&.filled']: {
			// 	transform: 'translate(12px, 20px) scale(1)',
			// },
		},
		errorOutline: {
			borderColor: red[500],

		},
		defaultContent: {
			position: 'relative',
			marginTop: marginTop,
			borderRadius: '4px',
			height: '100%',
			padding: padding,
			...(variant == 'filled') ? {
				paddingBottom: 2
			} : (variant == 'standard') ? {
				paddingTop: 0,
				paddingBottom: 5,
			} : {},
		},
		mediumContent: {
			...(variant == 'outlined') ? {
				paddingTop: 16,
				paddingLeft: 15,

			} : (variant == 'filled') ? {
				paddingTop: 20,
				paddingLeft: 12,

			} : {}
		},
		smallContent: {
			...(variant == 'outlined') ? {
				paddingTop: 16,
				paddingLeft: 12,

			} : (variant == 'filled') ? {
				paddingLeft: 12,

			} : {}
		},
		[customNotchedOutlineClassName]: {
			borderStyle: `solid ${grey[400]} 1px`
		},
		wrapper: {
			display: 'flex',
			position: 'relative',
			cursor: 'text',
			borderRadius: '4px',

			...(variant != 'outlined') ? {
				['&:hover']: {
					['&::before']: {
						left: 0,
						right: 0,
						bottom: 0,
						content: '""',
						position: 'absolute',
						transition: 'background-color 200ms cubic-bezier(0.0, 0, 0.2, 1) 0ms',
						borderBottom: `${variant == 'standard' ? 2 : 1}px solid ${isError ? red[500] : '#000000de'}`,
					},
				}
			} : {},

			[`&:hover .${customNotchedOutlineClassName}`]: {
				borderColor: `${isError ? red[500] : '#000000de'}`
			},
			[`&:hover .${customContentClassName}.filled`]: {
				backgroundColor: '#0000000d',
				transition: 'background-color 200ms cubic-bezier(0.0, 0, 0.2, 1) 0ms'
			},
		}
	}));

	const classes = useStyles();

	const { label, children, style, contentStyle, size, isShrink, placeholder, ...others } = props;
	return (
		<FormControl {...others} variant={variant} fullWidth={true} size={size} error={isError}
			style={{ ...{ position: 'relative', height: '100%' }, ...style }}
			className={classes.wrapper}>
			{
				label && <InputLabel ref={labelRef} shrink={isShrink} style={{ cursor: 'text' }}>{label}</InputLabel>
			}
			{
				variant == 'outlined' && (
					<NotchedOutline notched labelWidth={isShrink ? labelWidth : ''}
						className={clsx(customNotchedOutlineClassName, classes.defaultOutline, (isError ? classes.errorOutline : ''))} />
				)
			}
			<Box style={{ ...contentStyle }}
				className={clsx(
					classes.defaultContent,
					isError ? classes.errorOutline : '',
					customContentClassName, variant,
					classes[`${size}Content`]
				)}>
				{children}
				{
					placeholder && children == null && (isShrink != null || !label) && (
						<Typography>
							<Box component='span' className={clsx(classes.placeholder, /* variant, */ size, isShrink ? 'down' : '')}>{placeholder}</Box>
						</Typography>
					)
				}
			</Box>
		</FormControl>

	);
};