import React, { FC } from 'react';
import InputLabel from '@material-ui/core/InputLabel';
import NotchedOutline from '@material-ui/core/OutlinedInput/NotchedOutline';
import { createStyles, Theme, makeStyles } from '@material-ui/core/styles';
import { red, grey } from '@material-ui/core/colors';
import clsx from 'clsx';

interface IProps {
	label?: string
	isError?: boolean
}

const useStyles = makeStyles((theme: Theme) => createStyles({
	root: {
		position: 'relative',
		height: '100%'
	},
	defaultOutline: {
		borderColor: grey[400],

	},
	errorOutline: {
		borderColor: red[500],

	},
	errorLabel: {
		color: red[500],

	},
	content: {
		padding: '18.5px 14px',
		borderRadius: '4px'
	},
	inputLabel: {
		position: 'absolute',
		left: 0,
		top: 0,
		// slight alteration to spec spacing to match visual spec result
		transform: 'translate(0, 24px) scale(1)'
	}
}));

export const LabelOutline: FC<IProps> = ({ id, label, isError = false, children }) => {
	const [labelWidth, setLabelWidth] = React.useState(0);
	const labelRef = React.useRef(null);
	const classes = useStyles();

	React.useEffect(() => {
		const labelNode = labelRef.current;
		setLabelWidth(labelNode != null ? labelNode.offsetWidth : 0);
	}, [label]);

	return (
		<div style={{ position: 'relative', height: '100%' }}>
			<InputLabel ref={labelRef} htmlFor={id} variant="outlined"
				className={`${classes.inputLabel} ${isError ? classes.errorLabel : ''}`} shrink>
				{label}
			</InputLabel>
			<div className={classes.root}>
				<div id={id} className={classes.content}>
					{children}
					<NotchedOutline notched labelWidth={labelWidth} className={clsx(classes.defaultOutline, (isError ? classes.errorOutline : ''))} />
				</div>
			</div>
		</div>
	);
};
