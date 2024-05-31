import React, { Component, Fragment } from 'react';
import MDIcon from '@mdi/react';
import { mdiAccountCircle, mdiBackburger, mdiMenu } from '@mdi/js';
import store from 'servDir/redux/store';
import { AppBar, Toolbar, Typography, Tooltip, Grid } from '@material-ui/core';
import ColorButton from 'compDir/Button';
import { withStyles, createStyles, Theme } from '@material-ui/core/styles';
import { mdiLogout } from '@mdi/js';
import ActiveLink from 'compDir/ActiveLink';

export interface IState {
	showMenu?: boolean
}

interface IProps {
	onClickMenuBtn?(callBack: any): any
}

const useStyles = (theme: Theme) => createStyles({
	appBar: {
		justifyContent: 'center',
		height: 60,
		// backgroundColor: 'var(--header-color)',
	},
	title: {
		flex: 1,
		display: 'flex',
	},
});

class Header extends Component<IProps, IState> {
	// Set default properties's values
	public static defaultProps: Partial<IProps> = {
		onClickMenuBtn: () => true
	}

	// Set default state
	public state: IState = {
		showMenu: true,
	}

	clickMenuBtn = () => {
		this.props.onClickMenuBtn(() => {
			this.setState({ showMenu: !this.state.showMenu });
		});
	}

	public render(): React.ReactNode {
		const auth = store.getState().authReducer;
		const userInfo = auth.userInfo;
		const { classes } = this.props;

		return (
			<Fragment>

				<AppBar position="sticky" className={classes.appBar} style={{ backgroundColor: '#1565c0' }}>
					<Toolbar disableGutters={true} style={{ paddingInline: 15 }}>
						<Grid item>
							<Grid container spacing={2} wrap='nowrap' alignItems='center'>
								<Grid item>
									<ColorButton btnType="icon" btnColor="lightBlue" btnContrast={50}
										onClick={this.clickMenuBtn}>
										<MDIcon size={'18px'} path={this.state.showMenu ? mdiBackburger : mdiMenu} />
									</ColorButton>
								</Grid>
								<Grid item>
									<Typography variant="h5" noWrap>Ads Tracker</Typography>
								</Grid>
							</Grid>
						</Grid>
						<Grid item container spacing={3} style={{ flex: 1, marginInline: 70, padding: 0 }}>
							<Grid item>
								<ActiveLink minWidth={130} href="/" activeClassName="active">
									<a className='nav-link'>タスク一覧</a>
								</ActiveLink>
							</Grid>
							<Grid item>
								<ActiveLink minWidth={130} href="/forms/shared" activeClassName="active">
									<a className='nav-link'>スケジューラ</a>
								</ActiveLink>
							</Grid>
							{/* {
								userInfo.admin && (
									<Grid item>
										<ActiveLink href="/forms/admin" activeClassName="active">
											<a className='nav-link'>管理画面</a>
										</ActiveLink>
									</Grid>
								)
							} */}
						</Grid>
						<Grid item>
							<Grid container alignItems='center' wrap='nowrap'>
								<Grid container item alignItems='center' >
									<MDIcon style={{ marginRight: 5 }} size={'16px'} path={mdiAccountCircle} />
									<div>{`${userInfo.name} (${userInfo.uid})`}</div>
								</Grid>
								{
									process.env.NEXT_PUBLIC_NODE_ENV === 'development' && (

										<Grid item>
											<Tooltip title="ログアウト">
												<ColorButton btnType="icon" btnColor="lightBlue" btnContrast={50}
													style={{ marginLeft: 10 }}
													onClick={() => { window.location = '/login'; }}>
													<MDIcon size={'18px'} path={mdiLogout} />
												</ColorButton>
											</Tooltip>
										</Grid>
									)
								}
							</Grid>

						</Grid>
					</Toolbar>
				</AppBar>
			</Fragment>
		);
	}
}

export default withStyles(useStyles)(Header);



