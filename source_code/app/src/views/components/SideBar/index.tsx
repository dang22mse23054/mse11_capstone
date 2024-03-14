import React, { CSSProperties, FunctionComponent } from 'react';
// import styles from './styles.module.css';
import { makeStyles } from '@material-ui/core/styles';
import clsx from 'clsx';

import { Grid, Drawer} from '@material-ui/core';

interface IProps {
	isFullSidebar?: boolean
	style?: CSSProperties
}

const defaultStyle = {
	paddingTop: 3,
	display: 'flex',
	flexDirection: 'column',
	justifyContent: 'space-between'
};

const defaultProps: IProps = {
	isFullSidebar: true,
	style: defaultStyle
};

const initSidebar = (fullWidth = 300, smallWidth = 0) => {

	const useStyles = makeStyles((theme) => ({
		drawer: {
			width: fullWidth,
			flexShrink: 0,
			whiteSpace: 'nowrap',
			['& > .MuiPaper-root']: {
				color: '#fafafa !important'
			}
		},
		drawerPaper: {
			zIndex: 0,
			// padding: 20,
			// paddingLeft: 0,
			position: 'absolute',
			// height: `calc(100% - 30px)`
			backgroundColor: theme.palette.secondary.main

		},
		drawerOpen: {
			width: fullWidth,
			transition: theme.transitions.create('width', {
				easing: theme.transitions.easing.sharp,
				duration: theme.transitions.duration.enteringScreen,
			}),
		},
		drawerClose: {
			transition: theme.transitions.create('width', {
				easing: theme.transitions.easing.sharp,
				duration: theme.transitions.duration.leavingScreen,
			}),
			overflowX: 'hidden',
			width: smallWidth,
		}
	}));


	const SideBar: FunctionComponent<IProps> = (props: IProps) => {
		const classes = useStyles();
		const isFullSidebar = props.isFullSidebar;
		const style = {
			...defaultStyle,
			...props.style
		};

		return (
			<Grid item style={{ position: 'relative' }}>
				<Drawer variant="permanent"
					className={clsx(classes.drawer, {
						[classes.drawerOpen]: isFullSidebar,
						[classes.drawerClose]: !isFullSidebar,
					})}
					classes={{
						paper: clsx(classes.drawerPaper, {
							[classes.drawerOpen]: isFullSidebar,
							[classes.drawerClose]: !isFullSidebar,
						}),
					}}>
					<Grid {...(style ? { style } : {})}>{props.children}</Grid>
				</Drawer>
			</Grid>
		);
	};

	SideBar.defaultProps = defaultProps;

	return SideBar;
};

export { initSidebar };